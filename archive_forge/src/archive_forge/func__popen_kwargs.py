import logging
import os
import subprocess
import sys
from functools import lru_cache
from pkg_resources import resource_filename
from ._definitions import FNAME_PER_PLATFORM, get_platform
def _popen_kwargs(prevent_sigint=False):
    startupinfo = None
    preexec_fn = None
    creationflags = 0
    if sys.platform.startswith('win'):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    if prevent_sigint:
        if sys.platform.startswith('win'):
            creationflags = 512
        else:
            preexec_fn = os.setpgrp
    falsy = ('', '0', 'false', 'no')
    if os.getenv('IMAGEIO_FFMPEG_NO_PREVENT_SIGINT', '').lower() not in falsy:
        preexec_fn = None
    return {'startupinfo': startupinfo, 'creationflags': creationflags, 'preexec_fn': preexec_fn}