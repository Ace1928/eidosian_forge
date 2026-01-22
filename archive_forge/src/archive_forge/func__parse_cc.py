import platform
import sys
import os
import re
import shutil
import warnings
import traceback
import llvmlite.binding as ll
def _parse_cc(text):
    """
    Parse CUDA compute capability version string.
    """
    if not text:
        return None
    else:
        m = re.match('(\\d+)\\.(\\d+)', text)
        if not m:
            raise ValueError('Compute capability must be specified as a string of "major.minor" where major and minor are decimals')
        grp = m.groups()
        return (int(grp[0]), int(grp[1]))