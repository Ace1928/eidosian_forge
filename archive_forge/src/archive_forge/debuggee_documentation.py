import atexit
import ctypes
import os
import signal
import struct
import subprocess
import sys
import threading
from debugpy import launcher
from debugpy.common import log, messaging
from debugpy.launcher import output
List of functions that determine whether to pause after debuggee process exits.

Every function is invoked with exit code as the argument. If any of the functions
returns True, the launcher pauses and waits for user input before exiting.
