from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
class CopyOperation(PatchOperation):
    """ Copies an object property or an array element to a new location """

    def apply(self, obj):
        try:
            from_ptr = self.pointer_cls(self.operation['from'])
        except KeyError as ex:
            raise InvalidJsonPatch("The operation does not contain a 'from' member")
        subobj, part = from_ptr.to_last(obj)
        try:
            value = copy.deepcopy(subobj[part])
        except (KeyError, IndexError) as ex:
            raise JsonPatchConflict(str(ex))
        obj = AddOperation({'op': 'add', 'path': self.location, 'value': value}, pointer_cls=self.pointer_cls).apply(obj)
        return obj