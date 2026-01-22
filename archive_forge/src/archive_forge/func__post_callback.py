import asyncio
import inspect
import logging
import time
from functools import partial
import param
from ..util import edit_readonly, function_name
from .logging import LOG_PERIODIC_END, LOG_PERIODIC_START
from .state import curdoc_locked, set_curdoc, state
def _post_callback(self):
    cbname = function_name(self.callback)
    if self._doc and self.log:
        _periodic_logger.info(LOG_PERIODIC_END, id(self._doc), cbname, self.counter)
    if not self._background:
        with edit_readonly(state):
            state._busy_counter -= 1
    if self.timeout is not None:
        dt = (time.time() - self._start_time) * 1000
        if dt > self.timeout:
            self.stop()
    if self.counter == self.count:
        self.stop()