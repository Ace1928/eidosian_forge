import atexit
import os
import signal
import sys
import ovs.vlog
def _atexit_handler():
    _call_hooks(0)