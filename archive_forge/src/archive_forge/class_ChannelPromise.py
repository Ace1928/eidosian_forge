from __future__ import annotations
import inspect
import random
import threading
from collections import OrderedDict, UserDict
from collections.abc import Iterable, Mapping
from itertools import count, repeat
from time import sleep, time
from vine.utils import wraps
from .encoding import safe_repr as _safe_repr
class ChannelPromise:

    def __init__(self, contract):
        self.__contract__ = contract

    def __call__(self):
        try:
            return self.__value__
        except AttributeError:
            value = self.__value__ = self.__contract__()
            return value

    def __repr__(self):
        try:
            return repr(self.__value__)
        except AttributeError:
            return f'<promise: 0x{id(self.__contract__):x}>'