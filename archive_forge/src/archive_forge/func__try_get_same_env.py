import os
import sys
import hashlib
import filecmp
from collections import namedtuple
from shutil import which
from jedi.cache import memoize_method, time_cache
from jedi.inference.compiled.subprocess import CompiledSubprocess, \
import parso
def _try_get_same_env():
    env = SameEnvironment()
    if not os.path.basename(env.executable).lower().startswith('python'):
        if os.name == 'nt':
            checks = ('Scripts\\python.exe', 'python.exe')
        else:
            checks = ('bin/python%s.%s' % (sys.version_info[0], sys.version[1]), 'bin/python%s' % sys.version_info[0], 'bin/python')
        for check in checks:
            guess = os.path.join(sys.exec_prefix, check)
            if os.path.isfile(guess):
                return Environment(guess)
        return InterpreterEnvironment()
    return env