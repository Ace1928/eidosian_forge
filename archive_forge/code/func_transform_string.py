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
def transform_string(self, instring: str, *, debug: bool=False) -> str:
    """
        Extension to :class:`scan_string`, to modify matching text with modified tokens that may
        be returned from a parse action.  To use ``transform_string``, define a grammar and
        attach a parse action to it that modifies the returned token list.
        Invoking ``transform_string()`` on a target string will then scan for matches,
        and replace the matched text patterns according to the logic in the parse
        action.  ``transform_string()`` returns the resulting transformed string.

        Example::

            wd = Word(alphas)
            wd.set_parse_action(lambda toks: toks[0].title())

            print(wd.transform_string("now is the winter of our discontent made glorious summer by this sun of york."))

        prints::

            Now Is The Winter Of Our Discontent Made Glorious Summer By This Sun Of York.
        """
    out: List[str] = []
    lastE = 0
    self.keepTabs = True
    try:
        for t, s, e in self.scan_string(instring, debug=debug):
            out.append(instring[lastE:s])
            if t:
                if isinstance(t, ParseResults):
                    out += t.as_list()
                elif isinstance(t, Iterable) and (not isinstance(t, str_type)):
                    out.extend(t)
                else:
                    out.append(t)
            lastE = e
        out.append(instring[lastE:])
        out = [o for o in out if o]
        return ''.join([str(s) for s in _flatten(out)])
    except ParseBaseException as exc:
        if ParserElement.verbose_stacktrace:
            raise
        else:
            raise exc.with_traceback(None)