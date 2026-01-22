import logging
import sys
from types import TracebackType
from typing import Callable, Type
from pyquil.api._logger import logger
class UserMessageError(Exception):
    """
    A special class of error which only displays its traceback when the program
      is run in debug mode (eg, with `LOG_LEVEL=DEBUG`).

    The purpose of this is to improve the user experience, reducing noise in
      the case of errors for which the cause is known.
    """

    def __init__(self, message: str):
        if logger.level <= logging.DEBUG:
            super().__init__(message)
        else:
            self.message = message

    def __str__(self) -> str:
        if logger.level <= logging.DEBUG:
            return super(UserMessageError, self).__str__()
        else:
            return f'ERROR: {self.message}'

    def __repr__(self) -> str:
        if logger.level <= logging.DEBUG:
            return super(UserMessageError, self).__repr__()
        else:
            return f'UserMessageError: {str(self)}'