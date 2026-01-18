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
def wait_for_exit():
    try:
        code = process.wait()
        if sys.platform != 'win32' and code < 0:
            code &= 255
    except Exception:
        log.swallow_exception("Couldn't determine process exit code")
        code = -1
    log.info('{0} exited with code {1}', describe(), code)
    output.wait_for_remaining_output()
    should_wait = any((pred(code) for pred in wait_on_exit_predicates))
    try:
        launcher.channel.send_event('exited', {'exitCode': code})
    except Exception:
        pass
    if should_wait:
        _wait_for_user_input()
    try:
        launcher.channel.send_event('terminated')
    except Exception:
        pass