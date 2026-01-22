import logging
import os
import subprocess
import sys
from functools import lru_cache
from pkg_resources import resource_filename
from ._definitions import FNAME_PER_PLATFORM, get_platform
def _is_valid_exe(exe):
    cmd = [exe, '-version']
    try:
        with open(os.devnull, 'w') as null:
            subprocess.check_call(cmd, stdout=null, stderr=subprocess.STDOUT, **_popen_kwargs())
        return True
    except (OSError, ValueError, subprocess.CalledProcessError):
        return False