from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import unittest
from gc import collect
from gc import get_objects
from threading import active_count as active_thread_count
from time import sleep
from time import time
import psutil
from greenlet import greenlet as RawGreenlet
from greenlet import getcurrent
from greenlet._greenlet import get_pending_cleanup_count
from greenlet._greenlet import get_total_main_greenlets
from . import leakcheck
def assertScriptRaises(self, script_name, exitcodes=None):
    import subprocess
    with self.assertRaises(subprocess.CalledProcessError) as exc:
        output = self.run_script(script_name, show_output=False)
        __traceback_info__ = output
    if exitcodes is None:
        exitcodes = self.get_expected_returncodes_for_aborted_process()
    self.assertIn(exc.exception.returncode, exitcodes)
    return exc.exception