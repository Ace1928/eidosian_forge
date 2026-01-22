import os
import sys
from os.path import pardir, realpath
def _getuserbase():
    env_base = os.environ.get('PYTHONUSERBASE', None)
    if env_base:
        return env_base
    if sys.platform in {'emscripten', 'vxworks', 'wasi'}:
        return None

    def joinuser(*args):
        return os.path.expanduser(os.path.join(*args))
    if os.name == 'nt':
        base = os.environ.get('APPDATA') or '~'
        return joinuser(base, 'Python')
    if sys.platform == 'darwin' and sys._framework:
        return joinuser('~', 'Library', sys._framework, f'{sys.version_info[0]}.{sys.version_info[1]}')
    return joinuser('~', '.local')