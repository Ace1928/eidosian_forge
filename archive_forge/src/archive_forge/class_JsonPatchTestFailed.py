from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
class JsonPatchTestFailed(JsonPatchException, AssertionError):
    """ A Test operation failed """