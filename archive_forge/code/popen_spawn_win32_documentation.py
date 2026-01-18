import os
import msvcrt
import signal
import sys
import _winapi
from .context import reduction, get_spawning_popen, set_spawning_popen
from . import spawn
from . import util

    Start a subprocess to run the code of a process object
    