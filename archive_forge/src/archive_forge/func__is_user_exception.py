import sys
from ._basic import MatchesRegex
from ._higherorder import AfterPreproccessing
from ._impl import (
def _is_user_exception(exc):
    return isinstance(exc, Exception)