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
def _workaround(self, name, *args, **kwargs):
    """
        TODO Currently we're passing slice objects around. This should not
        happen. They are also the only unhashable objects that we're passing
        around.
        """
    if args and isinstance(args[0], slice):
        return self._subprocess.get_compiled_method_return(self.id, name, *args, **kwargs)
    return self._cached_results(name, *args, **kwargs)