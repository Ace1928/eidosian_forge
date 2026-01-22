from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
class OleDirectoryEntry:
    """
    OLE2 Directory Entry pointing to a stream or a storage
    """
    STRUCT_DIRENTRY = '<64sHBBIII16sIQQIII'
    DIRENTRY_SIZE = 128
    assert struct.calcsize(STRUCT_DIRENTRY) == DIRENTRY_SIZE

    def __init__(self, entry, sid, ole_file):
        """
        Constructor for an OleDirectoryEntry object.
        Parses a 128-bytes entry from the OLE Directory stream.

        :param bytes entry: bytes string (must be 128 bytes long)
        :param int sid: index of this directory entry in the OLE file directory
        :param OleFileIO ole_file: OleFileIO object containing this directory entry
        """
        self.sid = sid
        self.olefile = ole_file
        self.kids = []
        self.kids_dict = {}
        self.used = False
        self.name_raw, self.namelength, self.entry_type, self.color, self.sid_left, self.sid_right, self.sid_child, clsid, self.dwUserFlags, self.createTime, self.modifyTime, self.isectStart, self.sizeLow, self.sizeHigh = struct.unpack(OleDirectoryEntry.STRUCT_DIRENTRY, entry)
        if self.entry_type not in [STGTY_ROOT, STGTY_STORAGE, STGTY_STREAM, STGTY_EMPTY]:
            ole_file._raise_defect(DEFECT_INCORRECT, 'unhandled OLE storage type')
        if self.entry_type == STGTY_ROOT and sid != 0:
            ole_file._raise_defect(DEFECT_INCORRECT, 'duplicate OLE root entry')
        if sid == 0 and self.entry_type != STGTY_ROOT:
            ole_file._raise_defect(DEFECT_INCORRECT, 'incorrect OLE root entry')
        if self.namelength > 64:
            ole_file._raise_defect(DEFECT_INCORRECT, 'incorrect DirEntry name length >64 bytes')
            self.namelength = 64
        self.name_utf16 = self.name_raw[:self.namelength - 2]
        self.name = ole_file._decode_utf16_str(self.name_utf16)
        log.debug('DirEntry SID=%d: %s' % (self.sid, repr(self.name)))
        log.debug(' - type: %d' % self.entry_type)
        log.debug(' - sect: %Xh' % self.isectStart)
        log.debug(' - SID left: %d, right: %d, child: %d' % (self.sid_left, self.sid_right, self.sid_child))
        if ole_file.sectorsize == 512:
            if self.sizeHigh != 0 and self.sizeHigh != 4294967295:
                log.debug('sectorsize=%d, sizeLow=%d, sizeHigh=%d (%X)' % (ole_file.sectorsize, self.sizeLow, self.sizeHigh, self.sizeHigh))
                ole_file._raise_defect(DEFECT_UNSURE, 'incorrect OLE stream size')
            self.size = self.sizeLow
        else:
            self.size = self.sizeLow + (long(self.sizeHigh) << 32)
        log.debug(' - size: %d (sizeLow=%d, sizeHigh=%d)' % (self.size, self.sizeLow, self.sizeHigh))
        self.clsid = _clsid(clsid)
        if self.entry_type == STGTY_STORAGE and self.size != 0:
            ole_file._raise_defect(DEFECT_POTENTIAL, 'OLE storage with size>0')
        self.is_minifat = False
        if self.entry_type in (STGTY_ROOT, STGTY_STREAM) and self.size > 0:
            if self.size < ole_file.minisectorcutoff and self.entry_type == STGTY_STREAM:
                self.is_minifat = True
            else:
                self.is_minifat = False
            ole_file._check_duplicate_stream(self.isectStart, self.is_minifat)
        self.sect_chain = None

    def build_sect_chain(self, ole_file):
        """
        Build the sector chain for a stream (from the FAT or the MiniFAT)

        :param OleFileIO ole_file: OleFileIO object containing this directory entry
        :return: nothing
        """
        if self.sect_chain:
            return
        if self.entry_type not in (STGTY_ROOT, STGTY_STREAM) or self.size == 0:
            return
        self.sect_chain = list()
        if self.is_minifat and (not ole_file.minifat):
            ole_file.loadminifat()
        next_sect = self.isectStart
        while next_sect != ENDOFCHAIN:
            self.sect_chain.append(next_sect)
            if self.is_minifat:
                next_sect = ole_file.minifat[next_sect]
            else:
                next_sect = ole_file.fat[next_sect]

    def build_storage_tree(self):
        """
        Read and build the red-black tree attached to this OleDirectoryEntry
        object, if it is a storage.
        Note that this method builds a tree of all subentries, so it should
        only be called for the root object once.
        """
        log.debug('build_storage_tree: SID=%d - %s - sid_child=%d' % (self.sid, repr(self.name), self.sid_child))
        if self.sid_child != NOSTREAM:
            self.append_kids(self.sid_child)
            self.kids.sort()

    def append_kids(self, child_sid):
        """
        Walk through red-black tree of children of this directory entry to add
        all of them to the kids list. (recursive method)

        :param child_sid: index of child directory entry to use, or None when called
            first time for the root. (only used during recursion)
        """
        log.debug('append_kids: child_sid=%d' % child_sid)
        if child_sid == NOSTREAM:
            return
        if child_sid < 0 or child_sid >= len(self.olefile.direntries):
            self.olefile._raise_defect(DEFECT_INCORRECT, 'OLE DirEntry index out of range')
        else:
            child = self.olefile._load_direntry(child_sid)
            log.debug('append_kids: child_sid=%d - %s - sid_left=%d, sid_right=%d, sid_child=%d' % (child.sid, repr(child.name), child.sid_left, child.sid_right, child.sid_child))
            if child.used:
                self.olefile._raise_defect(DEFECT_INCORRECT, 'OLE Entry referenced more than once')
                return
            child.used = True
            self.append_kids(child.sid_left)
            name_lower = child.name.lower()
            if name_lower in self.kids_dict:
                self.olefile._raise_defect(DEFECT_INCORRECT, 'Duplicate filename in OLE storage')
            self.kids.append(child)
            self.kids_dict[name_lower] = child
            self.append_kids(child.sid_right)
            child.build_storage_tree()

    def __eq__(self, other):
        """Compare entries by name"""
        return self.name == other.name

    def __lt__(self, other):
        """Compare entries by name"""
        return self.name < other.name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)

    def dump(self, tab=0):
        """Dump this entry, and all its subentries (for debug purposes only)"""
        TYPES = ['(invalid)', '(storage)', '(stream)', '(lockbytes)', '(property)', '(root)']
        try:
            type_name = TYPES[self.entry_type]
        except IndexError:
            type_name = '(UNKNOWN)'
        print(' ' * tab + repr(self.name), type_name, end=' ')
        if self.entry_type in (STGTY_STREAM, STGTY_ROOT):
            print(self.size, 'bytes', end=' ')
        print()
        if self.entry_type in (STGTY_STORAGE, STGTY_ROOT) and self.clsid:
            print(' ' * tab + '{%s}' % self.clsid)
        for kid in self.kids:
            kid.dump(tab + 2)

    def getmtime(self):
        """
        Return modification time of a directory entry.

        :returns: None if modification time is null, a python datetime object
            otherwise (UTC timezone)

        new in version 0.26
        """
        if self.modifyTime == 0:
            return None
        return filetime2datetime(self.modifyTime)

    def getctime(self):
        """
        Return creation time of a directory entry.

        :returns: None if modification time is null, a python datetime object
            otherwise (UTC timezone)

        new in version 0.26
        """
        if self.createTime == 0:
            return None
        return filetime2datetime(self.createTime)