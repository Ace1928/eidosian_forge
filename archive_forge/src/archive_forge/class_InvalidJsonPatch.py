from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
class InvalidJsonPatch(JsonPatchException):
    """ Raised if an invalid JSON Patch is created """