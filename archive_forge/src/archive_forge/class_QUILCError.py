import logging
import sys
from types import TracebackType
from typing import Callable, Type
from pyquil.api._logger import logger
class QUILCError(ApiError):

    def __init__(self, server_status: str):
        explanation = '\nQUILC returned the above error. This could be due to a bug in the server or a\nbug in your code. If you suspect this to be a bug in pyQuil or Rigetti Forest,\nthen please describe the problem in a GitHub issue at:\n    https://github.com/rigetti/pyquil/issues'
        super(QUILCError, self).__init__(server_status, explanation)