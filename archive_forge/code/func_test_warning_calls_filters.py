from NumPy.
import os
from pathlib import Path
import ast
import tokenize
import scipy
import pytest
@pytest.mark.slow
def test_warning_calls_filters(warning_calls):
    bad_filters, bad_stacklevels = warning_calls
    allowed_filters = (os.path.join('datasets', '_fetchers.py'), os.path.join('datasets', '__init__.py'), os.path.join('optimize', '_optimize.py'), os.path.join('optimize', '_constraints.py'), os.path.join('signal', '_ltisys.py'), os.path.join('sparse', '__init__.py'), os.path.join('stats', '_discrete_distns.py'), os.path.join('stats', '_continuous_distns.py'), os.path.join('stats', '_binned_statistic.py'), os.path.join('_lib', '_util.py'))
    bad_filters = [item for item in bad_filters if item.split(':')[0] not in allowed_filters]
    if bad_filters:
        raise AssertionError('warning ignore filter should not be used, instead, use\nnumpy.testing.suppress_warnings (in tests only);\nfound in:\n    {}'.format('\n    '.join(bad_filters)))