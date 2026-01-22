import enum
import json
import os
import re
import typing as t
from collections import abc
from collections import deque
from random import choice
from random import randrange
from threading import Lock
from types import CodeType
from urllib.parse import quote_from_bytes
import markupsafe
def _postinit(self) -> None:
    self._popleft = self._queue.popleft
    self._pop = self._queue.pop
    self._remove = self._queue.remove
    self._wlock = Lock()
    self._append = self._queue.append