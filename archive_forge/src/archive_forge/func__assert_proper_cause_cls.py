import contextlib
import re
import sys
def _assert_proper_cause_cls(exception, cause_cls):
    """assert that any exception we're catching does not have a __context__
    without a __cause__, and that __suppress_context__ is never set.

    Python 3 will report nested as exceptions as "during the handling of
    error X, error Y occurred". That's not what we want to do. We want
    these exceptions in a cause chain.

    """
    assert isinstance(exception.__cause__, cause_cls), 'Exception %r was correctly raised but has cause %r, which does not have the expected cause type %r.' % (exception, exception.__cause__, cause_cls)