import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
class XCHierarchicalElement(XCObject):
    """Abstract base for PBXGroup and PBXFileReference.  Not represented in a
  project file."""
    _schema = XCObject._schema.copy()
    _schema.update({'comments': [0, str, 0, 0], 'fileEncoding': [0, str, 0, 0], 'includeInIndex': [0, int, 0, 0], 'indentWidth': [0, int, 0, 0], 'lineEnding': [0, int, 0, 0], 'sourceTree': [0, str, 0, 1, '<group>'], 'tabWidth': [0, int, 0, 0], 'usesTabs': [0, int, 0, 0], 'wrapsLines': [0, int, 0, 0]})

    def __init__(self, properties=None, id=None, parent=None):
        XCObject.__init__(self, properties, id, parent)
        if 'path' in self._properties and 'name' not in self._properties:
            path = self._properties['path']
            name = posixpath.basename(path)
            if name != '' and path != name:
                self.SetProperty('name', name)
        if 'path' in self._properties and ('sourceTree' not in self._properties or self._properties['sourceTree'] == '<group>'):
            source_tree, path = SourceTreeAndPathFromPath(self._properties['path'])
            if source_tree is not None:
                self._properties['sourceTree'] = source_tree
            if path is not None:
                self._properties['path'] = path
            if source_tree is not None and path is None and ('name' not in self._properties):
                del self._properties['path']
                self._properties['name'] = source_tree

    def Name(self):
        if 'name' in self._properties:
            return self._properties['name']
        elif 'path' in self._properties:
            return self._properties['path']
        else:
            return None

    def Hashables(self):
        """Custom hashables for XCHierarchicalElements.

    XCHierarchicalElements are special.  Generally, their hashes shouldn't
    change if the paths don't change.  The normal XCObject implementation of
    Hashables adds a hashable for each object, which means that if
    the hierarchical structure changes (possibly due to changes caused when
    TakeOverOnlyChild runs and encounters slight changes in the hierarchy),
    the hashes will change.  For example, if a project file initially contains
    a/b/f1 and a/b becomes collapsed into a/b, f1 will have a single parent
    a/b.  If someone later adds a/f2 to the project file, a/b can no longer be
    collapsed, and f1 winds up with parent b and grandparent a.  That would
    be sufficient to change f1's hash.

    To counteract this problem, hashables for all XCHierarchicalElements except
    for the main group (which has neither a name nor a path) are taken to be
    just the set of path components.  Because hashables are inherited from
    parents, this provides assurance that a/b/f1 has the same set of hashables
    whether its parent is b or a/b.

    The main group is a special case.  As it is permitted to have no name or
    path, it is permitted to use the standard XCObject hash mechanism.  This
    is not considered a problem because there can be only one main group.
    """
        if self == self.PBXProjectAncestor()._properties['mainGroup']:
            return XCObject.Hashables(self)
        hashables = []
        if 'name' in self._properties:
            hashables.append(self.__class__.__name__ + '.name')
            hashables.append(self._properties['name'])
        path = self.PathFromSourceTreeAndPath()
        if path is not None:
            components = path.split(posixpath.sep)
            for component in components:
                hashables.append(self.__class__.__name__ + '.path')
                hashables.append(component)
        hashables.extend(self._hashables)
        return hashables

    def Compare(self, other):
        valid_class_types = {PBXFileReference: 'file', PBXGroup: 'group', PBXVariantGroup: 'file'}
        self_type = valid_class_types[self.__class__]
        other_type = valid_class_types[other.__class__]
        if self_type == other_type:
            return cmp(self.Name(), other.Name())
        if self_type == 'group':
            return -1
        return 1

    def CompareRootGroup(self, other):
        order = ['Source', 'Intermediates', 'Projects', 'Frameworks', 'Products', 'Build']
        self_name = self.Name()
        other_name = other.Name()
        self_in = isinstance(self, PBXGroup) and self_name in order
        other_in = isinstance(self, PBXGroup) and other_name in order
        if not self_in and (not other_in):
            return self.Compare(other)
        if self_name in order and other_name not in order:
            return -1
        if other_name in order and self_name not in order:
            return 1
        self_index = order.index(self_name)
        other_index = order.index(other_name)
        if self_index < other_index:
            return -1
        if self_index > other_index:
            return 1
        return 0

    def PathFromSourceTreeAndPath(self):
        components = []
        if self._properties['sourceTree'] != '<group>':
            components.append('$(' + self._properties['sourceTree'] + ')')
        if 'path' in self._properties:
            components.append(self._properties['path'])
        if len(components) > 0:
            return posixpath.join(*components)
        return None

    def FullPath(self):
        xche = self
        path = None
        while isinstance(xche, XCHierarchicalElement) and (path is None or (not path.startswith('/') and (not path.startswith('$')))):
            this_path = xche.PathFromSourceTreeAndPath()
            if this_path is not None and path is not None:
                path = posixpath.join(this_path, path)
            elif this_path is not None:
                path = this_path
            xche = xche.parent
        return path