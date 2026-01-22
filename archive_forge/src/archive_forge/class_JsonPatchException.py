from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
class JsonPatchException(Exception):
    """Base Json Patch exception"""