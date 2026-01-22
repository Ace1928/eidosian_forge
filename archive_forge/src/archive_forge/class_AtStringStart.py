from collections import deque
import os
import typing
from typing import (
from abc import ABC, abstractmethod
from enum import Enum
import string
import copy
import warnings
import re
import sys
from collections.abc import Iterable
import traceback
import types
from operator import itemgetter
from functools import wraps
from threading import RLock
from pathlib import Path
from .util import (
from .exceptions import *
from .actions import *
from .results import ParseResults, _ParseResultsWithOffset
from .unicode import pyparsing_unicode
class AtStringStart(ParseElementEnhance):
    """Matches if expression matches at the beginning of the parse
    string::

        AtStringStart(Word(nums)).parse_string("123")
        # prints ["123"]

        AtStringStart(Word(nums)).parse_string("    123")
        # raises ParseException
    """

    def __init__(self, expr: Union[ParserElement, str]):
        super().__init__(expr)
        self.callPreparse = False

    def parseImpl(self, instring, loc, doActions=True):
        if loc != 0:
            raise ParseException(instring, loc, 'not found at string start')
        return super().parseImpl(instring, loc, doActions)