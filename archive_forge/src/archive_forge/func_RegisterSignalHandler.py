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
def RegisterSignalHandler(signal_num, handler, is_final_handler=False):
    """Registers a handler for signal signal_num.

  Unlike calling signal.signal():
    - This function can be called from any thread (and will cause the handler to
      be run by the main thread when the signal is received).
    - Handlers are cumulative: When a given signal is received, all registered
      handlers will be executed (with the exception that only the last handler
      to register with is_final_handler=True will be called).

  Handlers should make no ordering assumptions, other than that the last handler
  to register with is_final_handler=True will be called after all the other
  handlers.

  Args:
    signal_num: The signal number with which to associate handler.
    handler: The handler.
    is_final_handler: Bool indicator whether handler should be called last among
                      all the handlers for this signal_num. The last handler to
                      register this way survives; other handlers registered with
                      is_final_handler=True will not be called when the signal
                      is received.
  Raises:
    RuntimeError: if attempt is made to register a signal_num not in
        GetCaughtSignals.
  """
    if signal_num not in GetCaughtSignals():
        raise RuntimeError('Attempt to register handler (%s) for signal %d, which is not in GetCaughtSignals' % (handler, signal_num))
    if is_final_handler:
        _final_signal_handlers[signal_num] = handler
    else:
        _non_final_signal_handlers[signal_num].append(handler)