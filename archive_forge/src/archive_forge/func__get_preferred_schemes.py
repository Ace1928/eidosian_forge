import os
import sys
from os.path import pardir, realpath
def _get_preferred_schemes():
    if os.name == 'nt':
        return {'prefix': 'nt', 'home': 'posix_home', 'user': 'nt_user'}
    if sys.platform == 'darwin' and sys._framework:
        return {'prefix': 'posix_prefix', 'home': 'posix_home', 'user': 'osx_framework_user'}
    return {'prefix': 'posix_prefix', 'home': 'posix_home', 'user': 'posix_user'}