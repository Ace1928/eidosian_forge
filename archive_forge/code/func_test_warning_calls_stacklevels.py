from NumPy.
import os
from pathlib import Path
import ast
import tokenize
import scipy
import pytest
@pytest.mark.slow
@pytest.mark.xfail(reason='stacklevels currently missing')
def test_warning_calls_stacklevels(warning_calls):
    bad_filters, bad_stacklevels = warning_calls
    msg = ''
    if bad_filters:
        msg += 'warning ignore filter should not be used, instead, use\nnumpy.testing.suppress_warnings (in tests only);\nfound in:\n    {}'.format('\n    '.join(bad_filters))
        msg += '\n\n'
    if bad_stacklevels:
        msg += 'warnings should have an appropriate stacklevel:\n    {}'.format('\n    '.join(bad_stacklevels))
    if msg:
        raise AssertionError(msg)