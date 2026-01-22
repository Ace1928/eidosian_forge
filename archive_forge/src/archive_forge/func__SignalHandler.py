from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import re
import signal
import sys
import traceback
from gslib import metrics
from gslib.exception import ControlCException
from gslib.utils.constants import UTF8
from gslib.utils.system_util import IS_WINDOWS
def _SignalHandler(signal_num, cur_stack_frame):
    """Global signal handler.

  When a signal is caught we execute each registered handler for that signal.

  Args:
    signal_num: Signal that was caught.
    cur_stack_frame: Unused.
  """
    if signal_num in _non_final_signal_handlers:
        for handler in _non_final_signal_handlers[signal_num]:
            handler(signal_num, cur_stack_frame)
    if signal_num in _final_signal_handlers:
        _final_signal_handlers[signal_num](signal_num, cur_stack_frame)