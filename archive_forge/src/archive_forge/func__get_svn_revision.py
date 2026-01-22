import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def _get_svn_revision(self, path):
    """Return path's SVN revision number.
        """
    try:
        output = subprocess.check_output(['svnversion'], cwd=path)
    except (subprocess.CalledProcessError, OSError):
        pass
    else:
        m = re.match(b'(?P<revision>\\d+)', output)
        if m:
            return int(m.group('revision'))
    if sys.platform == 'win32' and os.environ.get('SVN_ASP_DOT_NET_HACK', None):
        entries = njoin(path, '_svn', 'entries')
    else:
        entries = njoin(path, '.svn', 'entries')
    if os.path.isfile(entries):
        with open(entries) as f:
            fstr = f.read()
        if fstr[:5] == '<?xml':
            m = re.search('revision="(?P<revision>\\d+)"', fstr)
            if m:
                return int(m.group('revision'))
        else:
            m = re.search('dir[\\n\\r]+(?P<revision>\\d+)', fstr)
            if m:
                return int(m.group('revision'))
    return None