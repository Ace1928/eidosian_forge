import logging
import queue
import sys
import threading
import time
from typing import TYPE_CHECKING, Optional, Tuple, Type, Union
from ..lib import tracelog
class ExceptionThread(threading.Thread):
    """Class to catch exceptions when running a thread."""
    __stopped: 'Event'
    __exception: Optional['ExceptionType']

    def __init__(self, stopped: 'Event') -> None:
        threading.Thread.__init__(self)
        self.__stopped = stopped
        self.__exception = None

    def _run(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        try:
            self._run()
        except Exception:
            self.__exception = sys.exc_info()
        finally:
            if self.__exception and self.__stopped:
                self.__stopped.set()

    def get_exception(self) -> Optional['ExceptionType']:
        return self.__exception