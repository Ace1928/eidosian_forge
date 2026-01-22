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
class AccessHandle:

    def __init__(self, subprocess, access, id_):
        self.access = access
        self._subprocess = subprocess
        self.id = id_

    def add_subprocess(self, subprocess):
        self._subprocess = subprocess

    def __repr__(self):
        try:
            detail = self.access
        except AttributeError:
            detail = '#' + str(self.id)
        return '<%s of %s>' % (self.__class__.__name__, detail)

    def __getstate__(self):
        return self.id

    def __setstate__(self, state):
        self.id = state

    def __getattr__(self, name):
        if name in ('id', 'access') or name.startswith('_'):
            raise AttributeError('Something went wrong with unpickling')
        return partial(self._workaround, name)

    def _workaround(self, name, *args, **kwargs):
        """
        TODO Currently we're passing slice objects around. This should not
        happen. They are also the only unhashable objects that we're passing
        around.
        """
        if args and isinstance(args[0], slice):
            return self._subprocess.get_compiled_method_return(self.id, name, *args, **kwargs)
        return self._cached_results(name, *args, **kwargs)

    @memoize_method
    def _cached_results(self, name, *args, **kwargs):
        return self._subprocess.get_compiled_method_return(self.id, name, *args, **kwargs)