import os
import sys
import hashlib
import filecmp
from collections import namedtuple
from shutil import which
from jedi.cache import memoize_method, time_cache
from jedi.inference.compiled.subprocess import CompiledSubprocess, \
import parso
def _get_virtual_env_from_var(env_var='VIRTUAL_ENV'):
    """Get virtualenv environment from VIRTUAL_ENV environment variable.

    It uses `safe=False` with ``create_environment``, because the environment
    variable is considered to be safe / controlled by the user solely.
    """
    var = os.environ.get(env_var)
    if var:
        if os.path.realpath(var) == os.path.realpath(sys.prefix):
            return _try_get_same_env()
        try:
            return create_environment(var, safe=False)
        except InvalidPythonEnvironment:
            pass