from abc import abstractmethod, ABC
import re
from contextlib import suppress
from typing import (
from types import ModuleType
import warnings
from .utils import classify, get_regexp_width, Serialize, logger
from .exceptions import UnexpectedCharacters, LexError, UnexpectedToken
from .grammar import TOKEN_DEFAULT_PRIORITY
from copy import copy
class CallChain:

    def __init__(self, callback1, callback2, cond):
        self.callback1 = callback1
        self.callback2 = callback2
        self.cond = cond

    def __call__(self, t):
        t2 = self.callback1(t)
        return self.callback2(t) if self.cond(t2) else t2