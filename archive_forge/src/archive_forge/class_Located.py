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
class Located(ParseElementEnhance):
    """
    Decorates a returned token with its starting and ending
    locations in the input string.

    This helper adds the following results names:

    - ``locn_start`` - location where matched expression begins
    - ``locn_end`` - location where matched expression ends
    - ``value`` - the actual parsed results

    Be careful if the input text contains ``<TAB>`` characters, you
    may want to call :class:`ParserElement.parse_with_tabs`

    Example::

        wd = Word(alphas)
        for match in Located(wd).search_string("ljsdf123lksdjjf123lkkjj1222"):
            print(match)

    prints::

        [0, ['ljsdf'], 5]
        [8, ['lksdjjf'], 15]
        [18, ['lkkjj'], 23]

    """

    def parseImpl(self, instring, loc, doActions=True):
        start = loc
        loc, tokens = self.expr._parse(instring, start, doActions, callPreParse=False)
        ret_tokens = ParseResults([start, tokens, loc])
        ret_tokens['locn_start'] = start
        ret_tokens['value'] = tokens
        ret_tokens['locn_end'] = loc
        if self.resultsName:
            return (loc, [ret_tokens])
        else:
            return (loc, ret_tokens)