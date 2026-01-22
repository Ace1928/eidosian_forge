import abc
import collections
from collections import abc as collections_abc
import copy
import functools
import logging
import warnings
import oslo_messaging as messaging
from oslo_utils import excutils
from oslo_utils import versionutils as vutils
from oslo_versionedobjects._i18n import _
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields as obj_fields
class ObjectListBase(collections_abc.Sequence):
    """Mixin class for lists of objects.

    This mixin class can be added as a base class for an object that
    is implementing a list of objects. It adds a single field of 'objects',
    which is the list store, and behaves like a list itself. It supports
    serialization of the list of objects automatically.
    """
    fields = {'objects': obj_fields.ListOfObjectsField('VersionedObject')}
    child_versions = {}

    def __init__(self, *args, **kwargs):
        super(ObjectListBase, self).__init__(*args, **kwargs)
        if 'objects' not in kwargs:
            self.objects = []
            self._changed_fields.discard('objects')

    def __len__(self):
        """List length."""
        return len(self.objects)

    def __getitem__(self, index):
        """List index access."""
        if isinstance(index, slice):
            new_obj = self.__class__()
            new_obj.objects = self.objects[index]
            new_obj.obj_reset_changes()
            new_obj._context = self._context
            return new_obj
        return self.objects[index]

    def sort(self, key=None, reverse=False):
        self.objects.sort(key=key, reverse=reverse)

    def obj_make_compatible(self, primitive, target_version):
        if self.child_versions:
            relationships = self.child_versions.items()
        else:
            try:
                relationships = self._obj_relationship_for('objects', target_version)
            except exception.ObjectActionError:
                relationships = {}
        try:
            if relationships:
                _get_subobject_version(target_version, relationships, lambda ver: _do_subobject_backport(ver, self, 'objects', primitive))
            else:
                _do_subobject_backport('1.0', self, 'objects', primitive)
        except exception.TargetBeforeSubobjectExistedException:
            del primitive['objects']

    def obj_what_changed(self):
        changes = set(self._changed_fields)
        for child in self.objects:
            if child.obj_what_changed():
                changes.add('objects')
        return changes

    def __add__(self, other):
        if self.__class__ == other.__class__ and list(self.__class__.fields.keys()) == ['objects']:
            return self.__class__(objects=self.objects + other.objects)
        else:
            raise TypeError("List Objects should be of the same type and only have an 'objects' field")

    def __radd__(self, other):
        if self.__class__ == other.__class__ and list(self.__class__.fields.keys()) == ['objects']:
            raise NotImplementedError('__radd__ is not implemented for objects of the same type')
        else:
            raise TypeError("List Objects should be of the same type and only have an 'objects' field")