import os
import msvcrt
import signal
import sys
import _winapi
from .context import reduction, get_spawning_popen, set_spawning_popen
from . import spawn
from . import util
def _path_eq(p1, p2):
    return p1 == p2 or os.path.normcase(p1) == os.path.normcase(p2)