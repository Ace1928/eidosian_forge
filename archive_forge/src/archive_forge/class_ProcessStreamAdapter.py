from abc import ABC, abstractmethod
import calendar
from collections import deque
from datetime import datetime, timedelta, tzinfo
from string import digits
import re
import time
import warnings
from git.util import IterableList, IterableObj, Actor
from typing import (
from git.types import Has_id_attribute, Literal  # , _T
class ProcessStreamAdapter:
    """Class wiring all calls to the contained Process instance.

    Use this type to hide the underlying process to provide access only to a specified
    stream. The process is usually wrapped into an AutoInterrupt class to kill
    it if the instance goes out of scope.
    """
    __slots__ = ('_proc', '_stream')

    def __init__(self, process: 'Popen', stream_name: str) -> None:
        self._proc = process
        self._stream: StringIO = getattr(process, stream_name)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._stream, attr)