from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
class JsonPatchConflict(JsonPatchException):
    """Raised if patch could not be applied due to conflict situation such as:
    - attempt to add object key when it already exists;
    - attempt to operate with nonexistence object key;
    - attempt to insert value to array at position beyond its size;
    - etc.
    """