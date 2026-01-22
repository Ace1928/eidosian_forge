from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
def _item_removed(self, path, key, item):
    new_op = RemoveOperation({'op': 'remove', 'path': _path_join(path, key)}, pointer_cls=self.pointer_cls)
    index = self.take_index(item, _ST_ADD)
    new_index = self.insert(new_op)
    if index is not None:
        op = index[2]
        added_item = op.pointer.to_last(self.dst_doc)[0]
        if type(added_item) == list:
            for v in self.iter_from(index):
                op.key = v._on_undo_add(op.path, op.key)
        self.remove(index)
        if new_op.location != op.location:
            new_op = MoveOperation({'op': 'move', 'from': new_op.location, 'path': op.location}, pointer_cls=self.pointer_cls)
            new_index[2] = new_op
        else:
            self.remove(new_index)
    else:
        self.store_index(item, new_index, _ST_REMOVE)