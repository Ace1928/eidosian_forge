import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
@property
def _log_prefix(self) -> str:
    return '>' * (len(self._stack) + 1) + ' '