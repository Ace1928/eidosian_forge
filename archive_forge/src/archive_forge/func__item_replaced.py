from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
def _item_replaced(self, path, key, item):
    self.insert(ReplaceOperation({'op': 'replace', 'path': _path_join(path, key), 'value': item}, pointer_cls=self.pointer_cls))