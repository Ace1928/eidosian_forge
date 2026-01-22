import sys
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, BinaryIO, Dict, Optional, cast
from zope.interface import Interface
from twisted.logger import (
from twisted.logger._global import LogBeginner
from twisted.logger._legacy import publishToNewObserver as _publishNew
from twisted.python import context, failure, reflect, util
from twisted.python.threadable import synchronize
class ILogObserver(Interface):
    """
    An observer which can do something with log events.

    Given that most log observers are actually bound methods, it's okay to not
    explicitly declare provision of this interface.
    """

    def __call__(eventDict: EventDict) -> None:
        """
        Log an event.

        @param eventDict: A dictionary with arbitrary keys.  However, these
            keys are often available:
              - C{message}: A C{tuple} of C{str} containing messages to be
                logged.
              - C{system}: A C{str} which indicates the "system" which is
                generating this event.
              - C{isError}: A C{bool} indicating whether this event represents
                an error.
              - C{failure}: A L{failure.Failure} instance
              - C{why}: Used as header of the traceback in case of errors.
              - C{format}: A string format used in place of C{message} to
                customize the event.  The intent is for the observer to format
                a message by doing something like C{format % eventDict}.
        """