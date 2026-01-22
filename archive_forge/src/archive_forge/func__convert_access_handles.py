import collections
import os
import sys
import queue
import subprocess
import traceback
import weakref
from functools import partial
from threading import Thread
from jedi._compatibility import pickle_dump, pickle_load
from jedi import debug
from jedi.cache import memoize_method
from jedi.inference.compiled.subprocess import functions
from jedi.inference.compiled.access import DirectObjectAccess, AccessPath, \
from jedi.api.exceptions import InternalError
def _convert_access_handles(self, obj):
    if isinstance(obj, SignatureParam):
        return SignatureParam(*self._convert_access_handles(tuple(obj)))
    elif isinstance(obj, tuple):
        return tuple((self._convert_access_handles(o) for o in obj))
    elif isinstance(obj, list):
        return [self._convert_access_handles(o) for o in obj]
    elif isinstance(obj, AccessHandle):
        try:
            obj = self.get_access_handle(obj.id)
        except KeyError:
            obj.add_subprocess(self)
            self.set_access_handle(obj)
    elif isinstance(obj, AccessPath):
        return AccessPath(self._convert_access_handles(obj.accesses))
    return obj