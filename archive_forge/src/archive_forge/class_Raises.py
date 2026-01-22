import sys
from ._basic import MatchesRegex
from ._higherorder import AfterPreproccessing
from ._impl import (
class Raises(Matcher):
    """Match if the matchee raises an exception when called.

    Exceptions which are not subclasses of Exception propagate out of the
    Raises.match call unless they are explicitly matched.
    """

    def __init__(self, exception_matcher=None):
        """Create a Raises matcher.

        :param exception_matcher: Optional validator for the exception raised
            by matchee. If supplied the exc_info tuple for the exception raised
            is passed into that matcher. If no exception_matcher is supplied
            then the simple fact of raising an exception is considered enough
            to match on.
        """
        self.exception_matcher = exception_matcher

    def match(self, matchee):
        try:
            result = matchee()
            return Mismatch(f'{matchee!r} returned {result!r}')
        except:
            exc_info = sys.exc_info()
            if self.exception_matcher:
                mismatch = self.exception_matcher.match(exc_info)
                if not mismatch:
                    del exc_info
                    return
            else:
                mismatch = None
            exception = exc_info[1]
            if _is_exception(exception) and (not _is_user_exception(exception)):
                del exc_info
                raise
            return mismatch

    def __str__(self):
        return 'Raises()'