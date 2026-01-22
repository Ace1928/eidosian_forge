import os
import msvcrt
import signal
import sys
import _winapi
from .context import reduction, get_spawning_popen, set_spawning_popen
from . import spawn
from . import util
def _close_handles(*handles):
    for handle in handles:
        _winapi.CloseHandle(handle)