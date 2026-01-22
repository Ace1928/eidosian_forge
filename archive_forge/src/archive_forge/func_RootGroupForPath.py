import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def RootGroupForPath(self, path):
    """Returns a PBXGroup child of this object to which path should be added.

    This method is intended to choose between SourceGroup and
    IntermediatesGroup on the basis of whether path is present in a source
    directory or an intermediates directory.  For the purposes of this
    determination, any path located within a derived file directory such as
    PROJECT_DERIVED_FILE_DIR is treated as being in an intermediates
    directory.

    The returned value is a two-element tuple.  The first element is the
    PBXGroup, and the second element specifies whether that group should be
    organized hierarchically (True) or as a single flat list (False).
    """
    source_tree_groups = {'DERIVED_FILE_DIR': (self.IntermediatesGroup, True), 'INTERMEDIATE_DIR': (self.IntermediatesGroup, True), 'PROJECT_DERIVED_FILE_DIR': (self.IntermediatesGroup, True), 'SHARED_INTERMEDIATE_DIR': (self.IntermediatesGroup, True)}
    source_tree, path = SourceTreeAndPathFromPath(path)
    if source_tree is not None and source_tree in source_tree_groups:
        group_func, hierarchical = source_tree_groups[source_tree]
        group = group_func()
        return (group, hierarchical)
    return (self.SourceGroup(), True)