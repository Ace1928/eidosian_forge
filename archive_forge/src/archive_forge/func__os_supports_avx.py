import platform
import sys
import os
import re
import shutil
import warnings
import traceback
import llvmlite.binding as ll
def _os_supports_avx():
    """
    Whether the current OS supports AVX, regardless of the CPU.

    This is necessary because the user may be running a very old Linux
    kernel (e.g. CentOS 5) on a recent CPU.
    """
    if not sys.platform.startswith('linux') or platform.machine() not in ('i386', 'i586', 'i686', 'x86_64'):
        return True
    try:
        f = open('/proc/cpuinfo', 'r')
    except OSError:
        return True
    with f:
        for line in f:
            head, _, body = line.partition(':')
            if head.strip() == 'flags' and 'avx' in body.split():
                return True
        else:
            return False