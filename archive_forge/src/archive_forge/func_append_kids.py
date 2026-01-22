from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
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