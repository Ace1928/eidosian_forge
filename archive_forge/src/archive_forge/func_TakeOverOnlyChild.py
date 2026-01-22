import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def TakeOverOnlyChild(self, recurse=False):
    """If this PBXGroup has only one child and it's also a PBXGroup, take
    it over by making all of its children this object's children.

    This function will continue to take over only children when those children
    are groups.  If there are three PBXGroups representing a, b, and c, with
    c inside b and b inside a, and a and b have no other children, this will
    result in a taking over both b and c, forming a PBXGroup for a/b/c.

    If recurse is True, this function will recurse into children and ask them
    to collapse themselves by taking over only children as well.  Assuming
    an example hierarchy with files at a/b/c/d1, a/b/c/d2, and a/b/c/d3/e/f
    (d1, d2, and f are files, the rest are groups), recursion will result in
    a group for a/b/c containing a group for d3/e.
    """
    while len(self._properties['children']) == 1 and self._properties['children'][0].__class__ == PBXGroup:
        child = self._properties['children'][0]
        old_properties = self._properties
        self._properties = child._properties
        self._children_by_path = child._children_by_path
        if 'sourceTree' not in self._properties or self._properties['sourceTree'] == '<group>':
            if 'path' in old_properties:
                if 'path' in self._properties:
                    self._properties['path'] = posixpath.join(old_properties['path'], self._properties['path'])
                else:
                    self._properties['path'] = old_properties['path']
            if 'sourceTree' in old_properties:
                self._properties['sourceTree'] = old_properties['sourceTree']
        if 'name' in old_properties and old_properties['name'] not in (None, self.Name()):
            self._properties['name'] = old_properties['name']
        if 'name' in self._properties and 'path' in self._properties and (self._properties['name'] == self._properties['path']):
            del self._properties['name']
        for child in self._properties['children']:
            child.parent = self
    if recurse:
        for child in self._properties['children']:
            if child.__class__ == PBXGroup:
                child.TakeOverOnlyChild(recurse)