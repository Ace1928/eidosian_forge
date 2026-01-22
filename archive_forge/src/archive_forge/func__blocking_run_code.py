import code
import greenlet
import logging
import signal
from curtsies.input import is_main_thread
def _blocking_run_code(self):
    try:
        unfinished = self.interp.runsource(self.source)
    except SystemExit as e:
        return SystemExitRequest(*e.args)
    return Unfinished() if unfinished else Done()