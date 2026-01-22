import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
class PBXGroup(XCHierarchicalElement):
    """
  Attributes:
    _children_by_path: Maps pathnames of children of this PBXGroup to the
      actual child XCHierarchicalElement objects.
    _variant_children_by_name_and_path: Maps (name, path) tuples of
      PBXVariantGroup children to the actual child PBXVariantGroup objects.
  """
    _schema = XCHierarchicalElement._schema.copy()
    _schema.update({'children': [1, XCHierarchicalElement, 1, 1, []], 'name': [0, str, 0, 0], 'path': [0, str, 0, 0]})

    def __init__(self, properties=None, id=None, parent=None):
        XCHierarchicalElement.__init__(self, properties, id, parent)
        self._children_by_path = {}
        self._variant_children_by_name_and_path = {}
        for child in self._properties.get('children', []):
            self._AddChildToDicts(child)

    def Hashables(self):
        hashables = XCHierarchicalElement.Hashables(self)
        for child in self._properties.get('children', []):
            child_name = child.Name()
            if child_name is not None:
                hashables.append(child_name)
        return hashables

    def HashablesForChild(self):
        return XCHierarchicalElement.Hashables(self)

    def _AddChildToDicts(self, child):
        child_path = child.PathFromSourceTreeAndPath()
        if child_path:
            if child_path in self._children_by_path:
                raise ValueError('Found multiple children with path ' + child_path)
            self._children_by_path[child_path] = child
        if isinstance(child, PBXVariantGroup):
            child_name = child._properties.get('name', None)
            key = (child_name, child_path)
            if key in self._variant_children_by_name_and_path:
                raise ValueError('Found multiple PBXVariantGroup children with ' + 'name ' + str(child_name) + ' and path ' + str(child_path))
            self._variant_children_by_name_and_path[key] = child

    def AppendChild(self, child):
        self.AppendProperty('children', child)
        self._AddChildToDicts(child)

    def GetChildByName(self, name):
        if 'children' not in self._properties:
            return None
        for child in self._properties['children']:
            if child.Name() == name:
                return child
        return None

    def GetChildByPath(self, path):
        if not path:
            return None
        if path in self._children_by_path:
            return self._children_by_path[path]
        return None

    def GetChildByRemoteObject(self, remote_object):
        if 'children' not in self._properties:
            return None
        for child in self._properties['children']:
            if not isinstance(child, PBXReferenceProxy):
                continue
            container_proxy = child._properties['remoteRef']
            if container_proxy._properties['remoteGlobalIDString'] == remote_object:
                return child
        return None

    def AddOrGetFileByPath(self, path, hierarchical):
        """Returns an existing or new file reference corresponding to path.

    If hierarchical is True, this method will create or use the necessary
    hierarchical group structure corresponding to path.  Otherwise, it will
    look in and create an item in the current group only.

    If an existing matching reference is found, it is returned, otherwise, a
    new one will be created, added to the correct group, and returned.

    If path identifies a directory by virtue of carrying a trailing slash,
    this method returns a PBXFileReference of "folder" type.  If path
    identifies a variant, by virtue of it identifying a file inside a directory
    with an ".lproj" extension, this method returns a PBXVariantGroup
    containing the variant named by path, and possibly other variants.  For
    all other paths, a "normal" PBXFileReference will be returned.
    """
        is_dir = False
        if path.endswith('/'):
            is_dir = True
        path = posixpath.normpath(path)
        if is_dir:
            path = path + '/'
        variant_name = None
        parent = posixpath.dirname(path)
        grandparent = posixpath.dirname(parent)
        parent_basename = posixpath.basename(parent)
        parent_root, parent_ext = posixpath.splitext(parent_basename)
        if parent_ext == '.lproj':
            variant_name = parent_root
        if grandparent == '':
            grandparent = None
        assert not is_dir or variant_name is None
        path_split = path.split(posixpath.sep)
        if len(path_split) == 1 or ((is_dir or variant_name is not None) and len(path_split) == 2) or (not hierarchical):
            if variant_name is None:
                file_ref = self.GetChildByPath(path)
                if file_ref is not None:
                    assert file_ref.__class__ == PBXFileReference
                else:
                    file_ref = PBXFileReference({'path': path})
                    self.AppendChild(file_ref)
            else:
                variant_group_name = posixpath.basename(path)
                variant_group_ref = self.AddOrGetVariantGroupByNameAndPath(variant_group_name, grandparent)
                variant_path = posixpath.sep.join(path_split[-2:])
                variant_ref = variant_group_ref.GetChildByPath(variant_path)
                if variant_ref is not None:
                    assert variant_ref.__class__ == PBXFileReference
                else:
                    variant_ref = PBXFileReference({'name': variant_name, 'path': variant_path})
                    variant_group_ref.AppendChild(variant_ref)
                file_ref = variant_group_ref
            return file_ref
        else:
            next_dir = path_split[0]
            group_ref = self.GetChildByPath(next_dir)
            if group_ref is not None:
                assert group_ref.__class__ == PBXGroup
            else:
                group_ref = PBXGroup({'path': next_dir})
                self.AppendChild(group_ref)
            return group_ref.AddOrGetFileByPath(posixpath.sep.join(path_split[1:]), hierarchical)

    def AddOrGetVariantGroupByNameAndPath(self, name, path):
        """Returns an existing or new PBXVariantGroup for name and path.

    If a PBXVariantGroup identified by the name and path arguments is already
    present as a child of this object, it is returned.  Otherwise, a new
    PBXVariantGroup with the correct properties is created, added as a child,
    and returned.

    This method will generally be called by AddOrGetFileByPath, which knows
    when to create a variant group based on the structure of the pathnames
    passed to it.
    """
        key = (name, path)
        if key in self._variant_children_by_name_and_path:
            variant_group_ref = self._variant_children_by_name_and_path[key]
            assert variant_group_ref.__class__ == PBXVariantGroup
            return variant_group_ref
        variant_group_properties = {'name': name}
        if path is not None:
            variant_group_properties['path'] = path
        variant_group_ref = PBXVariantGroup(variant_group_properties)
        self.AppendChild(variant_group_ref)
        return variant_group_ref

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

    def SortGroup(self):
        self._properties['children'] = sorted(self._properties['children'], key=cmp_to_key(lambda x, y: x.Compare(y)))
        for child in self._properties['children']:
            if isinstance(child, PBXGroup):
                child.SortGroup()