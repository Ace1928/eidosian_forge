import io
import os
import re
import sys
from ._core import Process
def _use_bsd_stat_format():
    try:
        return os.uname().sysname.lower() in ('freebsd', 'netbsd', 'dragonfly')
    except Exception:
        return False