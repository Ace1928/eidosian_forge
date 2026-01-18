import os
import textwrap
import pytest
from sklearn import __version__
from sklearn.utils._openmp_helpers import _openmp_parallelism_enabled
def test_openmp_parallelism_enabled():
    if os.getenv('SKLEARN_SKIP_OPENMP_TEST'):
        pytest.skip('test explicitly skipped (SKLEARN_SKIP_OPENMP_TEST)')
    base_url = 'dev' if __version__.endswith('.dev0') else 'stable'
    err_msg = textwrap.dedent('\n        This test fails because scikit-learn has been built without OpenMP.\n        This is not recommended since some estimators will run in sequential\n        mode instead of leveraging thread-based parallelism.\n\n        You can find instructions to build scikit-learn with OpenMP at this\n        address:\n\n            https://scikit-learn.org/{}/developers/advanced_installation.html\n\n        You can skip this test by setting the environment variable\n        SKLEARN_SKIP_OPENMP_TEST to any value.\n        ').format(base_url)
    assert _openmp_parallelism_enabled(), err_msg