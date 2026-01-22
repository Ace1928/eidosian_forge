from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
class MoveOperation(PatchOperation):
    """Moves an object property or an array element to a new location."""

    def apply(self, obj):
        try:
            if isinstance(self.operation['from'], self.pointer_cls):
                from_ptr = self.operation['from']
            else:
                from_ptr = self.pointer_cls(self.operation['from'])
        except KeyError as ex:
            raise InvalidJsonPatch("The operation does not contain a 'from' member")
        subobj, part = from_ptr.to_last(obj)
        try:
            value = subobj[part]
        except (KeyError, IndexError) as ex:
            raise JsonPatchConflict(str(ex))
        if self.pointer == from_ptr:
            return obj
        if isinstance(subobj, MutableMapping) and self.pointer.contains(from_ptr):
            raise JsonPatchConflict('Cannot move values into their own children')
        obj = RemoveOperation({'op': 'remove', 'path': self.operation['from']}, pointer_cls=self.pointer_cls).apply(obj)
        obj = AddOperation({'op': 'add', 'path': self.location, 'value': value}, pointer_cls=self.pointer_cls).apply(obj)
        return obj

    @property
    def from_path(self):
        from_ptr = self.pointer_cls(self.operation['from'])
        return '/'.join(from_ptr.parts[:-1])

    @property
    def from_key(self):
        from_ptr = self.pointer_cls(self.operation['from'])
        try:
            return int(from_ptr.parts[-1])
        except TypeError:
            return from_ptr.parts[-1]

    @from_key.setter
    def from_key(self, value):
        from_ptr = self.pointer_cls(self.operation['from'])
        from_ptr.parts[-1] = str(value)
        self.operation['from'] = from_ptr.path

    def _on_undo_remove(self, path, key):
        if self.from_path == path:
            if self.from_key >= key:
                self.from_key += 1
            else:
                key -= 1
        if self.path == path:
            if self.key > key:
                self.key += 1
            else:
                key += 1
        return key

    def _on_undo_add(self, path, key):
        if self.from_path == path:
            if self.from_key > key:
                self.from_key -= 1
            else:
                key -= 1
        if self.path == path:
            if self.key > key:
                self.key -= 1
            else:
                key += 1
        return key