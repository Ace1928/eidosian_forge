import abc
import ast
import bisect
import sys
import token
from ast import Module
from typing import Iterable, Iterator, List, Optional, Tuple, Any, cast, TYPE_CHECKING
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from .line_numbers import LineNumbers
from .util import (
def token_range(self, first_token, last_token, include_extra=False):
    """
    Yields all tokens in order from first_token through and including last_token. If
    include_extra is True, includes non-coding tokens such as tokenize.NL and .COMMENT.
    """
    for i in xrange(first_token.index, last_token.index + 1):
        if include_extra or not is_non_coding_token(self._tokens[i].type):
            yield self._tokens[i]