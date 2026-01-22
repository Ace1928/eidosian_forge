import os
import sys
import hashlib
import filecmp
from collections import namedtuple
from shutil import which
from jedi.cache import memoize_method, time_cache
from jedi.inference.compiled.subprocess import CompiledSubprocess, \
import parso
def _assert_safe(executable_path, safe):
    if safe and (not _is_safe(executable_path)):
        raise InvalidPythonEnvironment('The python binary is potentially unsafe.')