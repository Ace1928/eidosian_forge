import os
import subprocess as sp
from .compat import DEVNULL
from .config_defaults import FFMPEG_BINARY, IMAGEMAGICK_BINARY
def try_cmd(cmd):
    try:
        popen_params = {'stdout': sp.PIPE, 'stderr': sp.PIPE, 'stdin': DEVNULL}
        if os.name == 'nt':
            popen_params['creationflags'] = 134217728
        proc = sp.Popen(cmd, **popen_params)
        proc.communicate()
    except Exception as err:
        return (False, err)
    else:
        return (True, None)