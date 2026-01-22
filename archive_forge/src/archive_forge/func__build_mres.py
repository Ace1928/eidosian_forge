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
def _build_mres(self, terminals, max_size):
    postfix = '$' if self.match_whole else ''
    mres = []
    while terminals:
        pattern = u'|'.join((u'(?P<%s>%s)' % (t.name, t.pattern.to_regexp() + postfix) for t in terminals[:max_size]))
        if self.use_bytes:
            pattern = pattern.encode('latin-1')
        try:
            mre = self.re_.compile(pattern, self.g_regex_flags)
        except AssertionError:
            return self._build_mres(terminals, max_size // 2)
        mres.append(mre)
        terminals = terminals[max_size:]
    return mres