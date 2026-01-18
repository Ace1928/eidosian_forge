import sys
import os
import functools
import subprocess
import sysconfig
@functools.lru_cache()
def vars():
    if not enabled():
        return {}
    homebrew_prefix = subprocess.check_output(['brew', '--prefix'], text=True).strip()
    return locals()