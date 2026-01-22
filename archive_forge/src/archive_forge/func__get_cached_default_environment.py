import os
import sys
import hashlib
import filecmp
from collections import namedtuple
from shutil import which
from jedi.cache import memoize_method, time_cache
from jedi.inference.compiled.subprocess import CompiledSubprocess, \
import parso
@time_cache(seconds=10 * 60)
def _get_cached_default_environment():
    try:
        return get_default_environment()
    except InvalidPythonEnvironment:
        return InterpreterEnvironment()