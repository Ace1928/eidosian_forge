from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
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