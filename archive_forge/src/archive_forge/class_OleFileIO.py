from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
class OleFileIO:
    """
    OLE container object

    This class encapsulates the interface to an OLE 2 structured
    storage file.  Use the listdir and openstream methods to
    access the contents of this file.

    Object names are given as a list of strings, one for each subentry
    level.  The root entry should be omitted.  For example, the following
    code extracts all image streams from a Microsoft Image Composer file::

        with OleFileIO("fan.mic") as ole:

            for entry in ole.listdir():
                if entry[1:2] == "Image":
                    fin = ole.openstream(entry)
                    fout = open(entry[0:1], "wb")
                    while True:
                        s = fin.read(8192)
                        if not s:
                            break
                        fout.write(s)

    You can use the viewer application provided with the Python Imaging
    Library to view the resulting files (which happens to be standard
    TIFF files).
    """

    def __init__(self, filename=None, raise_defects=DEFECT_FATAL, write_mode=False, debug=False, path_encoding=DEFAULT_PATH_ENCODING):
        """
        Constructor for the OleFileIO class.

        :param filename: file to open.

            - if filename is a string smaller than 1536 bytes, it is the path
              of the file to open. (bytes or unicode string)
            - if filename is a string longer than 1535 bytes, it is parsed
              as the content of an OLE file in memory. (bytes type only)
            - if filename is a file-like object (with read, seek and tell methods),
              it is parsed as-is. The caller is responsible for closing it when done.

        :param raise_defects: minimal level for defects to be raised as exceptions.
            (use DEFECT_FATAL for a typical application, DEFECT_INCORRECT for a
            security-oriented application, see source code for details)

        :param write_mode: bool, if True the file is opened in read/write mode instead
            of read-only by default.

        :param debug: bool, set debug mode (deprecated, not used anymore)

        :param path_encoding: None or str, name of the codec to use for path
            names (streams and storages), or None for Unicode.
            Unicode by default on Python 3+, UTF-8 on Python 2.x.
            (new in olefile 0.42, was hardcoded to Latin-1 until olefile v0.41)
        """
        self._raise_defects_level = raise_defects
        self.parsing_issues = []
        self.write_mode = write_mode
        self.path_encoding = path_encoding
        self._filesize = None
        self.ministream = None
        self._used_streams_fat = []
        self._used_streams_minifat = []
        self.byte_order = None
        self.directory_fp = None
        self.direntries = None
        self.dll_version = None
        self.fat = None
        self.first_difat_sector = None
        self.first_dir_sector = None
        self.first_mini_fat_sector = None
        self.fp = None
        self.header_clsid = None
        self.header_signature = None
        self.metadata = None
        self.mini_sector_shift = None
        self.mini_sector_size = None
        self.mini_stream_cutoff_size = None
        self.minifat = None
        self.minifatsect = None
        self.minisectorcutoff = None
        self.minisectorsize = None
        self.ministream = None
        self.minor_version = None
        self.nb_sect = None
        self.num_difat_sectors = None
        self.num_dir_sectors = None
        self.num_fat_sectors = None
        self.num_mini_fat_sectors = None
        self.reserved1 = None
        self.reserved2 = None
        self.root = None
        self.sector_shift = None
        self.sector_size = None
        self.transaction_signature_number = None
        self.warn_if_not_closed = False
        self._we_opened_fp = False
        self._open_stack = None
        if filename:
            try:
                self.open(filename, write_mode=write_mode)
            except Exception:
                self._close(warn=False)
                raise

    def __del__(self):
        """Destructor, ensures all file handles are closed that we opened."""
        self._close(warn=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._close(warn=False)

    def _raise_defect(self, defect_level, message, exception_type=OleFileError):
        """
        This method should be called for any defect found during file parsing.
        It may raise an OleFileError exception according to the minimal level chosen
        for the OleFileIO object.

        :param defect_level: defect level, possible values are:

            - DEFECT_UNSURE    : a case which looks weird, but not sure it's a defect
            - DEFECT_POTENTIAL : a potential defect
            - DEFECT_INCORRECT : an error according to specifications, but parsing can go on
            - DEFECT_FATAL     : an error which cannot be ignored, parsing is impossible

        :param message: string describing the defect, used with raised exception.
        :param exception_type: exception class to be raised, OleFileError by default
        """
        if defect_level >= self._raise_defects_level:
            log.error(message)
            raise exception_type(message)
        else:
            self.parsing_issues.append((exception_type, message))
            log.warning(message)

    def _decode_utf16_str(self, utf16_str, errors='replace'):
        """
        Decode a string encoded in UTF-16 LE format, as found in the OLE
        directory or in property streams. Return a string encoded
        according to the path_encoding specified for the OleFileIO object.

        :param bytes utf16_str: bytes string encoded in UTF-16 LE format
        :param str errors: str, see python documentation for str.decode()
        :return: str, encoded according to path_encoding
        :rtype: str
        """
        unicode_str = utf16_str.decode('UTF-16LE', errors)
        if self.path_encoding:
            return unicode_str.encode(self.path_encoding, errors)
        else:
            return unicode_str

    def open(self, filename, write_mode=False):
        """
        Open an OLE2 file in read-only or read/write mode.
        Read and parse the header, FAT and directory.

        :param filename: string-like or file-like object, OLE file to parse

            - if filename is a string smaller than 1536 bytes, it is the path
              of the file to open. (bytes or unicode string)
            - if filename is a string longer than 1535 bytes, it is parsed
              as the content of an OLE file in memory. (bytes type only)
            - if filename is a file-like object (with read, seek and tell methods),
              it is parsed as-is. The caller is responsible for closing it when done

        :param write_mode: bool, if True the file is opened in read/write mode instead
            of read-only by default. (ignored if filename is not a path)
        """
        self.write_mode = write_mode
        if hasattr(filename, 'read'):
            self.fp = filename
        elif isinstance(filename, bytes) and len(filename) >= MINIMAL_OLEFILE_SIZE:
            self.fp = io.BytesIO(filename)
        else:
            if self.write_mode:
                mode = 'r+b'
            else:
                mode = 'rb'
            self.fp = open(filename, mode)
            self._we_opened_fp = True
            self._open_stack = traceback.extract_stack()
        filesize = 0
        self.fp.seek(0, os.SEEK_END)
        try:
            filesize = self.fp.tell()
        finally:
            self.fp.seek(0)
        self._filesize = filesize
        log.debug('File size: %d bytes (%Xh)' % (self._filesize, self._filesize))
        self._used_streams_fat = []
        self._used_streams_minifat = []
        header = self.fp.read(512)
        if len(header) != 512 or header[:8] != MAGIC:
            log.debug('Magic = {!r} instead of {!r}'.format(header[:8], MAGIC))
            self._raise_defect(DEFECT_FATAL, 'not an OLE2 structured storage file', NotOleFileError)
        fmt_header = '<8s16sHHHHHHLLLLLLLLLL'
        header_size = struct.calcsize(fmt_header)
        log.debug('fmt_header size = %d, +FAT = %d' % (header_size, header_size + 109 * 4))
        header1 = header[:header_size]
        self.header_signature, self.header_clsid, self.minor_version, self.dll_version, self.byte_order, self.sector_shift, self.mini_sector_shift, self.reserved1, self.reserved2, self.num_dir_sectors, self.num_fat_sectors, self.first_dir_sector, self.transaction_signature_number, self.mini_stream_cutoff_size, self.first_mini_fat_sector, self.num_mini_fat_sectors, self.first_difat_sector, self.num_difat_sectors = struct.unpack(fmt_header, header1)
        log.debug(struct.unpack(fmt_header, header1))
        if self.header_signature != MAGIC:
            self._raise_defect(DEFECT_FATAL, 'incorrect OLE signature')
        if self.header_clsid != bytearray(16):
            self._raise_defect(DEFECT_INCORRECT, 'incorrect CLSID in OLE header')
        log.debug('Minor Version = %d' % self.minor_version)
        log.debug('DLL Version   = %d (expected: 3 or 4)' % self.dll_version)
        if self.dll_version not in [3, 4]:
            self._raise_defect(DEFECT_INCORRECT, 'incorrect DllVersion in OLE header')
        log.debug('Byte Order    = %X (expected: FFFE)' % self.byte_order)
        if self.byte_order != 65534:
            self._raise_defect(DEFECT_INCORRECT, 'incorrect ByteOrder in OLE header')
        self.sector_size = 2 ** self.sector_shift
        log.debug('Sector Size   = %d bytes (expected: 512 or 4096)' % self.sector_size)
        if self.sector_size not in [512, 4096]:
            self._raise_defect(DEFECT_INCORRECT, 'incorrect sector_size in OLE header')
        if self.dll_version == 3 and self.sector_size != 512 or (self.dll_version == 4 and self.sector_size != 4096):
            self._raise_defect(DEFECT_INCORRECT, 'sector_size does not match DllVersion in OLE header')
        self.mini_sector_size = 2 ** self.mini_sector_shift
        log.debug('MiniFAT Sector Size   = %d bytes (expected: 64)' % self.mini_sector_size)
        if self.mini_sector_size not in [64]:
            self._raise_defect(DEFECT_INCORRECT, 'incorrect mini_sector_size in OLE header')
        if self.reserved1 != 0 or self.reserved2 != 0:
            self._raise_defect(DEFECT_INCORRECT, 'incorrect OLE header (non-null reserved bytes)')
        log.debug('Number of Directory sectors = %d' % self.num_dir_sectors)
        if self.sector_size == 512 and self.num_dir_sectors != 0:
            self._raise_defect(DEFECT_INCORRECT, 'incorrect number of directory sectors in OLE header')
        log.debug('Number of FAT sectors = %d' % self.num_fat_sectors)
        log.debug('First Directory sector  = %Xh' % self.first_dir_sector)
        log.debug('Transaction Signature Number    = %d' % self.transaction_signature_number)
        if self.transaction_signature_number != 0:
            self._raise_defect(DEFECT_POTENTIAL, 'incorrect OLE header (transaction_signature_number>0)')
        log.debug('Mini Stream cutoff size = %Xh (expected: 1000h)' % self.mini_stream_cutoff_size)
        if self.mini_stream_cutoff_size != 4096:
            self._raise_defect(DEFECT_INCORRECT, 'incorrect mini_stream_cutoff_size in OLE header')
            log.warning('Fixing the mini_stream_cutoff_size to 4096 (mandatory value) instead of %d' % self.mini_stream_cutoff_size)
            self.mini_stream_cutoff_size = 4096
        log.debug('First MiniFAT sector      = %Xh' % self.first_mini_fat_sector)
        log.debug('Number of MiniFAT sectors = %d' % self.num_mini_fat_sectors)
        log.debug('First DIFAT sector        = %Xh' % self.first_difat_sector)
        log.debug('Number of DIFAT sectors   = %d' % self.num_difat_sectors)
        self.nb_sect = (filesize + self.sector_size - 1) // self.sector_size - 1
        log.debug('Maximum number of sectors in the file: %d (%Xh)' % (self.nb_sect, self.nb_sect))
        self.header_clsid = _clsid(header[8:24])
        self.sectorsize = self.sector_size
        self.minisectorsize = self.mini_sector_size
        self.minisectorcutoff = self.mini_stream_cutoff_size
        self._check_duplicate_stream(self.first_dir_sector)
        if self.num_mini_fat_sectors:
            self._check_duplicate_stream(self.first_mini_fat_sector)
        if self.num_difat_sectors:
            self._check_duplicate_stream(self.first_difat_sector)
        self.loadfat(header)
        self.loaddirectory(self.first_dir_sector)
        self.minifatsect = self.first_mini_fat_sector

    def close(self):
        """
        close the OLE file, release the file object if we created it ourselves.

        Leaves the file handle open if it was provided by the caller.
        """
        self._close(warn=False)

    def _close(self, warn=False):
        """Implementation of close() with internal arg `warn`."""
        if self._we_opened_fp:
            if warn and self.warn_if_not_closed:
                warnings.warn(OleFileIONotClosed(self._open_stack))
            self.fp.close()
            self._we_opened_fp = False

    def _check_duplicate_stream(self, first_sect, minifat=False):
        """
        Checks if a stream has not been already referenced elsewhere.
        This method should only be called once for each known stream, and only
        if stream size is not null.

        :param first_sect: int, index of first sector of the stream in FAT
        :param minifat: bool, if True, stream is located in the MiniFAT, else in the FAT
        """
        if minifat:
            log.debug('_check_duplicate_stream: sect=%Xh in MiniFAT' % first_sect)
            used_streams = self._used_streams_minifat
        else:
            log.debug('_check_duplicate_stream: sect=%Xh in FAT' % first_sect)
            if first_sect in (DIFSECT, FATSECT, ENDOFCHAIN, FREESECT):
                return
            used_streams = self._used_streams_fat
        if first_sect in used_streams:
            self._raise_defect(DEFECT_INCORRECT, 'Stream referenced twice')
        else:
            used_streams.append(first_sect)

    def dumpfat(self, fat, firstindex=0):
        """
        Display a part of FAT in human-readable form for debugging purposes
        """
        VPL = 8
        fatnames = {FREESECT: '..free..', ENDOFCHAIN: '[ END. ]', FATSECT: 'FATSECT ', DIFSECT: 'DIFSECT '}
        nbsect = len(fat)
        nlines = (nbsect + VPL - 1) // VPL
        print('index', end=' ')
        for i in range(VPL):
            print('%8X' % i, end=' ')
        print()
        for l in range(nlines):
            index = l * VPL
            print('%6X:' % (firstindex + index), end=' ')
            for i in range(index, index + VPL):
                if i >= nbsect:
                    break
                sect = fat[i]
                aux = sect & 4294967295
                if aux in fatnames:
                    name = fatnames[aux]
                elif sect == i + 1:
                    name = '    --->'
                else:
                    name = '%8X' % sect
                print(name, end=' ')
            print()

    def dumpsect(self, sector, firstindex=0):
        """
        Display a sector in a human-readable form, for debugging purposes
        """
        VPL = 8
        tab = array.array(UINT32, sector)
        if sys.byteorder == 'big':
            tab.byteswap()
        nbsect = len(tab)
        nlines = (nbsect + VPL - 1) // VPL
        print('index', end=' ')
        for i in range(VPL):
            print('%8X' % i, end=' ')
        print()
        for l in range(nlines):
            index = l * VPL
            print('%6X:' % (firstindex + index), end=' ')
            for i in range(index, index + VPL):
                if i >= nbsect:
                    break
                sect = tab[i]
                name = '%8X' % sect
                print(name, end=' ')
            print()

    def sect2array(self, sect):
        """
        convert a sector to an array of 32 bits unsigned integers,
        swapping bytes on big endian CPUs such as PowerPC (old Macs)
        """
        a = array.array(UINT32, sect)
        if sys.byteorder == 'big':
            a.byteswap()
        return a

    def loadfat_sect(self, sect):
        """
        Adds the indexes of the given sector to the FAT

        :param sect: string containing the first FAT sector, or array of long integers
        :returns: index of last FAT sector.
        """
        if isinstance(sect, array.array):
            fat1 = sect
        else:
            fat1 = self.sect2array(sect)
            if log.isEnabledFor(logging.DEBUG):
                self.dumpsect(sect)
        isect = None
        for isect in fat1:
            isect = isect & 4294967295
            log.debug('isect = %X' % isect)
            if isect == ENDOFCHAIN or isect == FREESECT:
                log.debug('found end of sector chain')
                break
            s = self.getsect(isect)
            nextfat = self.sect2array(s)
            self.fat = self.fat + nextfat
        return isect

    def loadfat(self, header):
        """
        Load the FAT table.
        """
        log.debug('Loading the FAT table, starting with the 1st sector after the header')
        sect = header[76:512]
        log.debug('len(sect)=%d, so %d integers' % (len(sect), len(sect) // 4))
        self.fat = array.array(UINT32)
        self.loadfat_sect(sect)
        if self.num_difat_sectors != 0:
            log.debug('DIFAT is used, because file size > 6.8MB.')
            if self.num_fat_sectors <= 109:
                self._raise_defect(DEFECT_INCORRECT, 'incorrect DIFAT, not enough sectors')
            if self.first_difat_sector >= self.nb_sect:
                self._raise_defect(DEFECT_FATAL, 'incorrect DIFAT, first index out of range')
            log.debug('DIFAT analysis...')
            nb_difat_sectors = self.sectorsize // 4 - 1
            nb_difat = (self.num_fat_sectors - 109 + nb_difat_sectors - 1) // nb_difat_sectors
            log.debug('nb_difat = %d' % nb_difat)
            if self.num_difat_sectors != nb_difat:
                raise IOError('incorrect DIFAT')
            isect_difat = self.first_difat_sector
            for i in iterrange(nb_difat):
                log.debug('DIFAT block %d, sector %X' % (i, isect_difat))
                sector_difat = self.getsect(isect_difat)
                difat = self.sect2array(sector_difat)
                if log.isEnabledFor(logging.DEBUG):
                    self.dumpsect(sector_difat)
                self.loadfat_sect(difat[:nb_difat_sectors])
                isect_difat = difat[nb_difat_sectors]
                log.debug('next DIFAT sector: %X' % isect_difat)
            if isect_difat not in [ENDOFCHAIN, FREESECT]:
                raise IOError('incorrect end of DIFAT')
        else:
            log.debug('No DIFAT, because file size < 6.8MB.')
        if len(self.fat) > self.nb_sect:
            log.debug('len(fat)=%d, shrunk to nb_sect=%d' % (len(self.fat), self.nb_sect))
            self.fat = self.fat[:self.nb_sect]
        log.debug('FAT references %d sectors / Maximum %d sectors in file' % (len(self.fat), self.nb_sect))
        if log.isEnabledFor(logging.DEBUG):
            log.debug('\nFAT:')
            self.dumpfat(self.fat)

    def loadminifat(self):
        """
        Load the MiniFAT table.
        """
        stream_size = self.num_mini_fat_sectors * self.sector_size
        nb_minisectors = (self.root.size + self.mini_sector_size - 1) // self.mini_sector_size
        used_size = nb_minisectors * 4
        log.debug('loadminifat(): minifatsect=%d, nb FAT sectors=%d, used_size=%d, stream_size=%d, nb MiniSectors=%d' % (self.minifatsect, self.num_mini_fat_sectors, used_size, stream_size, nb_minisectors))
        if used_size > stream_size:
            self._raise_defect(DEFECT_INCORRECT, 'OLE MiniStream is larger than MiniFAT')
        s = self._open(self.minifatsect, stream_size, force_FAT=True).read()
        self.minifat = self.sect2array(s)
        log.debug('MiniFAT shrunk from %d to %d sectors' % (len(self.minifat), nb_minisectors))
        self.minifat = self.minifat[:nb_minisectors]
        log.debug('loadminifat(): len=%d' % len(self.minifat))
        if log.isEnabledFor(logging.DEBUG):
            log.debug('\nMiniFAT:')
            self.dumpfat(self.minifat)

    def getsect(self, sect):
        """
        Read given sector from file on disk.

        :param sect: int, sector index
        :returns: a string containing the sector data.
        """
        try:
            self.fp.seek(self.sectorsize * (sect + 1))
        except Exception:
            log.debug('getsect(): sect=%X, seek=%d, filesize=%d' % (sect, self.sectorsize * (sect + 1), self._filesize))
            self._raise_defect(DEFECT_FATAL, 'OLE sector index out of range')
        sector = self.fp.read(self.sectorsize)
        if len(sector) != self.sectorsize:
            log.debug('getsect(): sect=%X, read=%d, sectorsize=%d' % (sect, len(sector), self.sectorsize))
            self._raise_defect(DEFECT_FATAL, 'incomplete OLE sector')
        return sector

    def write_sect(self, sect, data, padding=b'\x00'):
        """
        Write given sector to file on disk.

        :param sect: int, sector index
        :param data: bytes, sector data
        :param padding: single byte, padding character if data < sector size
        """
        if not isinstance(data, bytes):
            raise TypeError('write_sect: data must be a bytes string')
        if not isinstance(padding, bytes) or len(padding) != 1:
            raise TypeError('write_sect: padding must be a bytes string of 1 char')
        try:
            self.fp.seek(self.sectorsize * (sect + 1))
        except Exception:
            log.debug('write_sect(): sect=%X, seek=%d, filesize=%d' % (sect, self.sectorsize * (sect + 1), self._filesize))
            self._raise_defect(DEFECT_FATAL, 'OLE sector index out of range')
        if len(data) < self.sectorsize:
            data += padding * (self.sectorsize - len(data))
        elif len(data) > self.sectorsize:
            raise ValueError('Data is larger than sector size')
        self.fp.write(data)

    def _write_mini_sect(self, fp_pos, data, padding=b'\x00'):
        """
        Write given sector to file on disk.

        :param fp_pos: int, file position
        :param data: bytes, sector data
        :param padding: single byte, padding character if data < sector size
        """
        if not isinstance(data, bytes):
            raise TypeError('write_mini_sect: data must be a bytes string')
        if not isinstance(padding, bytes) or len(padding) != 1:
            raise TypeError('write_mini_sect: padding must be a bytes string of 1 char')
        try:
            self.fp.seek(fp_pos)
        except Exception:
            log.debug('write_mini_sect(): fp_pos=%d, filesize=%d' % (fp_pos, self._filesize))
            self._raise_defect(DEFECT_FATAL, 'OLE sector index out of range')
        len_data = len(data)
        if len_data < self.mini_sector_size:
            data += padding * (self.mini_sector_size - len_data)
        if self.mini_sector_size < len_data:
            raise ValueError('Data is larger than sector size')
        self.fp.write(data)

    def loaddirectory(self, sect):
        """
        Load the directory.

        :param sect: sector index of directory stream.
        """
        log.debug('Loading the Directory:')
        self.directory_fp = self._open(sect, force_FAT=True)
        max_entries = self.directory_fp.size // 128
        log.debug('loaddirectory: size=%d, max_entries=%d' % (self.directory_fp.size, max_entries))
        self.direntries = [None] * max_entries
        root_entry = self._load_direntry(0)
        self.root = self.direntries[0]
        self.root.build_storage_tree()

    def _load_direntry(self, sid):
        """
        Load a directory entry from the directory.
        This method should only be called once for each storage/stream when
        loading the directory.

        :param sid: index of storage/stream in the directory.
        :returns: a OleDirectoryEntry object

        :exception OleFileError: if the entry has always been referenced.
        """
        if sid < 0 or sid >= len(self.direntries):
            self._raise_defect(DEFECT_FATAL, 'OLE directory index out of range')
        if self.direntries[sid] is not None:
            self._raise_defect(DEFECT_INCORRECT, 'double reference for OLE stream/storage')
            return self.direntries[sid]
        self.directory_fp.seek(sid * 128)
        entry = self.directory_fp.read(128)
        self.direntries[sid] = OleDirectoryEntry(entry, sid, self)
        return self.direntries[sid]

    def dumpdirectory(self):
        """
        Dump directory (for debugging only)
        """
        self.root.dump()

    def _open(self, start, size=UNKNOWN_SIZE, force_FAT=False):
        """
        Open a stream, either in FAT or MiniFAT according to its size.
        (openstream helper)

        :param start: index of first sector
        :param size: size of stream (or nothing if size is unknown)
        :param force_FAT: if False (default), stream will be opened in FAT or MiniFAT
            according to size. If True, it will always be opened in FAT.
        """
        log.debug('OleFileIO.open(): sect=%Xh, size=%d, force_FAT=%s' % (start, size, str(force_FAT)))
        if size < self.minisectorcutoff and (not force_FAT):
            if not self.ministream:
                self.loadminifat()
                size_ministream = self.root.size
                log.debug('Opening MiniStream: sect=%Xh, size=%d' % (self.root.isectStart, size_ministream))
                self.ministream = self._open(self.root.isectStart, size_ministream, force_FAT=True)
            return OleStream(fp=self.ministream, sect=start, size=size, offset=0, sectorsize=self.minisectorsize, fat=self.minifat, filesize=self.ministream.size, olefileio=self)
        else:
            return OleStream(fp=self.fp, sect=start, size=size, offset=self.sectorsize, sectorsize=self.sectorsize, fat=self.fat, filesize=self._filesize, olefileio=self)

    def _list(self, files, prefix, node, streams=True, storages=False):
        """
        listdir helper

        :param files: list of files to fill in
        :param prefix: current location in storage tree (list of names)
        :param node: current node (OleDirectoryEntry object)
        :param streams: bool, include streams if True (True by default) - new in v0.26
        :param storages: bool, include storages if True (False by default) - new in v0.26
            (note: the root storage is never included)
        """
        prefix = prefix + [node.name]
        for entry in node.kids:
            if entry.entry_type == STGTY_STORAGE:
                if storages:
                    files.append(prefix[1:] + [entry.name])
                self._list(files, prefix, entry, streams, storages)
            elif entry.entry_type == STGTY_STREAM:
                if streams:
                    files.append(prefix[1:] + [entry.name])
            else:
                self._raise_defect(DEFECT_INCORRECT, 'The directory tree contains an entry which is not a stream nor a storage.')

    def listdir(self, streams=True, storages=False):
        """
        Return a list of streams and/or storages stored in this file

        :param streams: bool, include streams if True (True by default) - new in v0.26
        :param storages: bool, include storages if True (False by default) - new in v0.26
            (note: the root storage is never included)
        :returns: list of stream and/or storage paths
        """
        files = []
        self._list(files, [], self.root, streams, storages)
        return files

    def _find(self, filename):
        """
        Returns directory entry of given filename. (openstream helper)
        Note: this method is case-insensitive.

        :param filename: path of stream in storage tree (except root entry), either:

            - a string using Unix path syntax, for example:
              'storage_1/storage_1.2/stream'
            - or a list of storage filenames, path to the desired stream/storage.
              Example: ['storage_1', 'storage_1.2', 'stream']

        :returns: sid of requested filename
        :exception IOError: if file not found
        """
        if isinstance(filename, basestring):
            filename = filename.split('/')
        node = self.root
        for name in filename:
            for kid in node.kids:
                if kid.name.lower() == name.lower():
                    break
            else:
                raise IOError('file not found')
            node = kid
        return node.sid

    def openstream(self, filename):
        """
        Open a stream as a read-only file object (BytesIO).
        Note: filename is case-insensitive.

        :param filename: path of stream in storage tree (except root entry), either:

            - a string using Unix path syntax, for example:
              'storage_1/storage_1.2/stream'
            - or a list of storage filenames, path to the desired stream/storage.
              Example: ['storage_1', 'storage_1.2', 'stream']

        :returns: file object (read-only)
        :exception IOError: if filename not found, or if this is not a stream.
        """
        sid = self._find(filename)
        entry = self.direntries[sid]
        if entry.entry_type != STGTY_STREAM:
            raise IOError('this file is not a stream')
        return self._open(entry.isectStart, entry.size)

    def _write_mini_stream(self, entry, data_to_write):
        if not entry.sect_chain:
            entry.build_sect_chain(self)
        nb_sectors = len(entry.sect_chain)
        if not self.root.sect_chain:
            self.root.build_sect_chain(self)
        block_size = self.sector_size // self.mini_sector_size
        for idx, sect in enumerate(entry.sect_chain):
            sect_base = sect // block_size
            sect_offset = sect % block_size
            fp_pos = (self.root.sect_chain[sect_base] + 1) * self.sector_size + sect_offset * self.mini_sector_size
            if idx < nb_sectors - 1:
                data_per_sector = data_to_write[idx * self.mini_sector_size:(idx + 1) * self.mini_sector_size]
            else:
                data_per_sector = data_to_write[idx * self.mini_sector_size:]
            self._write_mini_sect(fp_pos, data_per_sector)

    def write_stream(self, stream_name, data):
        """
        Write a stream to disk. For now, it is only possible to replace an
        existing stream by data of the same size.

        :param stream_name: path of stream in storage tree (except root entry), either:

            - a string using Unix path syntax, for example:
              'storage_1/storage_1.2/stream'
            - or a list of storage filenames, path to the desired stream/storage.
              Example: ['storage_1', 'storage_1.2', 'stream']

        :param data: bytes, data to be written, must be the same size as the original
            stream.
        """
        if not isinstance(data, bytes):
            raise TypeError('write_stream: data must be a bytes string')
        sid = self._find(stream_name)
        entry = self.direntries[sid]
        if entry.entry_type != STGTY_STREAM:
            raise IOError('this is not a stream')
        size = entry.size
        if size != len(data):
            raise ValueError('write_stream: data must be the same size as the existing stream')
        if size < self.minisectorcutoff and entry.entry_type != STGTY_ROOT:
            return self._write_mini_stream(entry=entry, data_to_write=data)
        sect = entry.isectStart
        nb_sectors = (size + (self.sectorsize - 1)) // self.sectorsize
        log.debug('nb_sectors = %d' % nb_sectors)
        for i in range(nb_sectors):
            if i < nb_sectors - 1:
                data_sector = data[i * self.sectorsize:(i + 1) * self.sectorsize]
                assert len(data_sector) == self.sectorsize
            else:
                data_sector = data[i * self.sectorsize:]
                log.debug('write_stream: size=%d sectorsize=%d data_sector=%Xh size%%sectorsize=%d' % (size, self.sectorsize, len(data_sector), size % self.sectorsize))
                assert len(data_sector) % self.sectorsize == size % self.sectorsize
            self.write_sect(sect, data_sector)
            try:
                sect = self.fat[sect]
            except IndexError:
                raise IOError('incorrect OLE FAT, sector index out of range')
        if sect != ENDOFCHAIN:
            raise IOError('incorrect last sector index in OLE stream')

    def get_type(self, filename):
        """
        Test if given filename exists as a stream or a storage in the OLE
        container, and return its type.

        :param filename: path of stream in storage tree. (see openstream for syntax)
        :returns: False if object does not exist, its entry type (>0) otherwise:

            - STGTY_STREAM: a stream
            - STGTY_STORAGE: a storage
            - STGTY_ROOT: the root entry
        """
        try:
            sid = self._find(filename)
            entry = self.direntries[sid]
            return entry.entry_type
        except Exception:
            return False

    def getclsid(self, filename):
        """
        Return clsid of a stream/storage.

        :param filename: path of stream/storage in storage tree. (see openstream for
            syntax)
        :returns: Empty string if clsid is null, a printable representation of the clsid otherwise

        new in version 0.44
        """
        sid = self._find(filename)
        entry = self.direntries[sid]
        return entry.clsid

    def getmtime(self, filename):
        """
        Return modification time of a stream/storage.

        :param filename: path of stream/storage in storage tree. (see openstream for
            syntax)
        :returns: None if modification time is null, a python datetime object
            otherwise (UTC timezone)

        new in version 0.26
        """
        sid = self._find(filename)
        entry = self.direntries[sid]
        return entry.getmtime()

    def getctime(self, filename):
        """
        Return creation time of a stream/storage.

        :param filename: path of stream/storage in storage tree. (see openstream for
            syntax)
        :returns: None if creation time is null, a python datetime object
            otherwise (UTC timezone)

        new in version 0.26
        """
        sid = self._find(filename)
        entry = self.direntries[sid]
        return entry.getctime()

    def exists(self, filename):
        """
        Test if given filename exists as a stream or a storage in the OLE
        container.
        Note: filename is case-insensitive.

        :param filename: path of stream in storage tree. (see openstream for syntax)
        :returns: True if object exist, else False.
        """
        try:
            sid = self._find(filename)
            return True
        except Exception:
            return False

    def get_size(self, filename):
        """
        Return size of a stream in the OLE container, in bytes.

        :param filename: path of stream in storage tree (see openstream for syntax)
        :returns: size in bytes (long integer)
        :exception IOError: if file not found
        :exception TypeError: if this is not a stream.
        """
        sid = self._find(filename)
        entry = self.direntries[sid]
        if entry.entry_type != STGTY_STREAM:
            raise TypeError('object is not an OLE stream')
        return entry.size

    def get_rootentry_name(self):
        """
        Return root entry name. Should usually be 'Root Entry' or 'R' in most
        implementations.
        """
        return self.root.name

    def getproperties(self, filename, convert_time=False, no_conversion=None):
        """
        Return properties described in substream.

        :param filename: path of stream in storage tree (see openstream for syntax)
        :param convert_time: bool, if True timestamps will be converted to Python datetime
        :param no_conversion: None or list of int, timestamps not to be converted
            (for example total editing time is not a real timestamp)

        :returns: a dictionary of values indexed by id (integer)
        """
        if no_conversion == None:
            no_conversion = []
        streampath = filename
        if not isinstance(streampath, str):
            streampath = '/'.join(streampath)
        fp = self.openstream(filename)
        data = {}
        try:
            s = fp.read(28)
            clsid = _clsid(s[8:24])
            s = fp.read(20)
            fmtid = _clsid(s[:16])
            fp.seek(i32(s, 16))
            s = b'****' + fp.read(i32(fp.read(4)) - 4)
            num_props = i32(s, 4)
        except BaseException as exc:
            msg = 'Error while parsing properties header in stream {}: {}'.format(repr(streampath), exc)
            self._raise_defect(DEFECT_INCORRECT, msg, type(exc))
            return data
        num_props = min(num_props, int(len(s) / 8))
        for i in iterrange(num_props):
            property_id = 0
            try:
                property_id = i32(s, 8 + i * 8)
                offset = i32(s, 12 + i * 8)
                property_type = i32(s, offset)
                vt_name = VT.get(property_type, 'UNKNOWN')
                log.debug('property id=%d: type=%d/%s offset=%X' % (property_id, property_type, vt_name, offset))
                value = self._parse_property(s, offset + 4, property_id, property_type, convert_time, no_conversion)
                data[property_id] = value
            except BaseException as exc:
                msg = 'Error while parsing property id %d in stream %s: %s' % (property_id, repr(streampath), exc)
                self._raise_defect(DEFECT_INCORRECT, msg, type(exc))
        return data

    def _parse_property(self, s, offset, property_id, property_type, convert_time, no_conversion):
        v = None
        if property_type <= VT_BLOB or property_type in (VT_CLSID, VT_CF):
            v, _ = self._parse_property_basic(s, offset, property_id, property_type, convert_time, no_conversion)
        elif property_type == VT_VECTOR | VT_VARIANT:
            log.debug('property_type == VT_VECTOR | VT_VARIANT')
            off = 4
            count = i32(s, offset)
            values = []
            for _ in range(count):
                property_type = i32(s, offset + off)
                v, sz = self._parse_property_basic(s, offset + off + 4, property_id, property_type, convert_time, no_conversion)
                values.append(v)
                off += sz + 4
            v = values
        elif property_type & VT_VECTOR:
            property_type_base = property_type & ~VT_VECTOR
            log.debug('property_type == VT_VECTOR | %s' % VT.get(property_type_base, 'UNKNOWN'))
            off = 4
            count = i32(s, offset)
            values = []
            for _ in range(count):
                v, sz = self._parse_property_basic(s, offset + off, property_id, property_type & ~VT_VECTOR, convert_time, no_conversion)
                values.append(v)
                off += sz
            v = values
        else:
            log.debug('property id=%d: type=%d not implemented in parser yet' % (property_id, property_type))
        return v

    def _parse_property_basic(self, s, offset, property_id, property_type, convert_time, no_conversion):
        value = None
        size = 0
        if property_type == VT_I2:
            value = i16(s, offset)
            if value >= 32768:
                value = value - 65536
            size = 2
        elif property_type == VT_UI2:
            value = i16(s, offset)
            size = 2
        elif property_type in (VT_I4, VT_INT, VT_ERROR):
            value = i32(s, offset)
            size = 4
        elif property_type in (VT_UI4, VT_UINT):
            value = i32(s, offset)
            size = 4
        elif property_type in (VT_BSTR, VT_LPSTR):
            count = i32(s, offset)
            value = s[offset + 4:offset + 4 + count - 1]
            value = value.replace(b'\x00', b'')
            size = 4 + count
        elif property_type == VT_BLOB:
            count = i32(s, offset)
            value = s[offset + 4:offset + 4 + count]
            size = 4 + count
        elif property_type == VT_LPWSTR:
            count = i32(s, offset + 4)
            value = self._decode_utf16_str(s[offset + 4:offset + 4 + count * 2])
            size = 4 + count * 2
        elif property_type == VT_FILETIME:
            value = long(i32(s, offset)) + (long(i32(s, offset + 4)) << 32)
            if convert_time and property_id not in no_conversion:
                log.debug('Converting property #%d to python datetime, value=%d=%fs' % (property_id, value, float(value) / 10000000))
                _FILETIME_null_date = datetime.datetime(1601, 1, 1, 0, 0, 0)
                log.debug('timedelta days=%d' % (value // (10 * 1000000 * 3600 * 24)))
                value = _FILETIME_null_date + datetime.timedelta(microseconds=value // 10)
            else:
                value = value // 10000000
            size = 8
        elif property_type == VT_UI1:
            value = i8(s[offset])
            size = 1
        elif property_type == VT_CLSID:
            value = _clsid(s[offset:offset + 16])
            size = 16
        elif property_type == VT_CF:
            count = i32(s, offset)
            value = s[offset + 4:offset + 4 + count]
            size = 4 + count
        elif property_type == VT_BOOL:
            value = bool(i16(s, offset))
            size = 2
        else:
            value = None
            log.debug('property id=%d: type=%d not implemented in parser yet' % (property_id, property_type))
        return (value, size)

    def get_metadata(self):
        """
        Parse standard properties streams, return an OleMetadata object
        containing all the available metadata.
        (also stored in the metadata attribute of the OleFileIO object)

        new in version 0.25
        """
        self.metadata = OleMetadata()
        self.metadata.parse_properties(self)
        return self.metadata

    def get_userdefined_properties(self, filename, convert_time=False, no_conversion=None):
        """
        Return properties described in substream.

        :param filename: path of stream in storage tree (see openstream for syntax)
        :param convert_time: bool, if True timestamps will be converted to Python datetime
        :param no_conversion: None or list of int, timestamps not to be converted
            (for example total editing time is not a real timestamp)

        :returns: a dictionary of values indexed by id (integer)
        """
        FMTID_USERDEFINED_PROPERTIES = _clsid(b'\x05\xd5\xcd\xd5\x9c.\x1b\x10\x93\x97\x08\x00+,\xf9\xae')
        if no_conversion == None:
            no_conversion = []
        streampath = filename
        if not isinstance(streampath, str):
            streampath = '/'.join(streampath)
        fp = self.openstream(filename)
        data = []
        s = fp.read(28)
        clsid = _clsid(s[8:24])
        sections_count = i32(s, 24)
        section_file_pointers = []
        try:
            for i in range(sections_count):
                s = fp.read(20)
                fmtid = _clsid(s[:16])
                if fmtid == FMTID_USERDEFINED_PROPERTIES:
                    file_pointer = i32(s, 16)
                    fp.seek(file_pointer)
                    s = b'****' + fp.read(i32(fp.read(4)) - 4)
                    num_props = i32(s, 4)
                    PropertyIdentifierAndOffset = s[8:8 + 8 * num_props]
                    index = 8 + 8 * num_props
                    entry_count = i32(s[index:index + 4])
                    index += 4
                    for i in range(entry_count):
                        identifier = s[index:index + 4]
                        str_size = i32(s[index + 4:index + 8])
                        string = s[index + 8:index + 8 + str_size].decode('utf_8').strip('\x00')
                        data.append({'property_name': string, 'value': None})
                        index = index + 8 + str_size
                    num_props = min(num_props, int(len(s) / 8))
                    for i in iterrange(2, num_props):
                        property_id = 0
                        try:
                            property_id = i32(s, 8 + i * 8)
                            offset = i32(s, 12 + i * 8)
                            property_type = i32(s, offset)
                            vt_name = VT.get(property_type, 'UNKNOWN')
                            log.debug('property id=%d: type=%d/%s offset=%X' % (property_id, property_type, vt_name, offset))
                            if property_type == VT_I2:
                                value = i16(s, offset + 4)
                                if value >= 32768:
                                    value = value - 65536
                            elif property_type == 1:
                                str_size = i32(s, offset + 8)
                                value = s[offset + 12:offset + 12 + str_size - 1]
                            elif property_type == VT_UI2:
                                value = i16(s, offset + 4)
                            elif property_type in (VT_I4, VT_INT, VT_ERROR):
                                value = i32(s, offset + 4)
                            elif property_type in (VT_UI4, VT_UINT):
                                value = i32(s, offset + 4)
                            elif property_type in (VT_BSTR, VT_LPSTR):
                                count = i32(s, offset + 4)
                                value = s[offset + 8:offset + 8 + count - 1]
                                value = value.replace(b'\x00', b'')
                            elif property_type == VT_BLOB:
                                count = i32(s, offset + 4)
                                value = s[offset + 8:offset + 8 + count]
                            elif property_type == VT_LPWSTR:
                                count = i32(s, offset + 4)
                                value = self._decode_utf16_str(s[offset + 8:offset + 8 + count * 2])
                            elif property_type == VT_FILETIME:
                                value = long(i32(s, offset + 4)) + (long(i32(s, offset + 8)) << 32)
                                if convert_time and property_id not in no_conversion:
                                    log.debug('Converting property #%d to python datetime, value=%d=%fs' % (property_id, value, float(value) / 10000000))
                                    _FILETIME_null_date = datetime.datetime(1601, 1, 1, 0, 0, 0)
                                    log.debug('timedelta days=%d' % (value // (10 * 1000000 * 3600 * 24)))
                                    value = _FILETIME_null_date + datetime.timedelta(microseconds=value // 10)
                                else:
                                    value = value // 10000000
                            elif property_type == VT_UI1:
                                value = i8(s[offset + 4])
                            elif property_type == VT_CLSID:
                                value = _clsid(s[offset + 4:offset + 20])
                            elif property_type == VT_CF:
                                count = i32(s, offset + 4)
                                value = s[offset + 8:offset + 8 + count]
                            elif property_type == VT_BOOL:
                                value = bool(i16(s, offset + 4))
                            else:
                                value = None
                                log.debug('property id=%d: type=%d not implemented in parser yet' % (property_id, property_type))
                            data[i - 2]['value'] = value
                        except BaseException as exc:
                            msg = 'Error while parsing property id %d in stream %s: %s' % (property_id, repr(streampath), exc)
                            self._raise_defect(DEFECT_INCORRECT, msg, type(exc))
        except BaseException as exc:
            msg = 'Error while parsing properties header in stream %s: %s' % (repr(streampath), exc)
            self._raise_defect(DEFECT_INCORRECT, msg, type(exc))
            return data
        return data