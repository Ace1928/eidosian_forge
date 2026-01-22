from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
class PatchOperation(object):
    """A single operation inside a JSON Patch."""

    def __init__(self, operation, pointer_cls=JsonPointer):
        self.pointer_cls = pointer_cls
        if not operation.__contains__('path'):
            raise InvalidJsonPatch("Operation must have a 'path' member")
        if isinstance(operation['path'], self.pointer_cls):
            self.location = operation['path'].path
            self.pointer = operation['path']
        else:
            self.location = operation['path']
            try:
                self.pointer = self.pointer_cls(self.location)
            except TypeError as ex:
                raise InvalidJsonPatch("Invalid 'path'")
        self.operation = operation

    def apply(self, obj):
        """Abstract method that applies a patch operation to the specified object."""
        raise NotImplementedError('should implement the patch operation.')

    def __hash__(self):
        return hash(frozenset(self.operation.items()))

    def __eq__(self, other):
        if not isinstance(other, PatchOperation):
            return False
        return self.operation == other.operation

    def __ne__(self, other):
        return not self == other

    @property
    def path(self):
        return '/'.join(self.pointer.parts[:-1])

    @property
    def key(self):
        try:
            return int(self.pointer.parts[-1])
        except ValueError:
            return self.pointer.parts[-1]

    @key.setter
    def key(self, value):
        self.pointer.parts[-1] = str(value)
        self.location = self.pointer.path
        self.operation['path'] = self.location