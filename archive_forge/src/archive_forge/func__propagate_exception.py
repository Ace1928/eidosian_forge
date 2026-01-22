import os
import socket
from contextlib import closing
import logging
import queue
import threading
from typing import Optional
import numpy as np
from ray.air.constants import _ERROR_REPORT_TIMEOUT
def _propagate_exception(self, e: BaseException):
    try:
        self._error_queue.put(e, block=True, timeout=_ERROR_REPORT_TIMEOUT)
    except queue.Full:
        logger.critical('Runner Thread was unable to report error to main function runner thread. This means a previous error was not processed. This should never happen.')