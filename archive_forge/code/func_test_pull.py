import pytest
import textwrap
import types
import warnings
from itertools import product
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface_lib._rinterface_capi
import rpy2.robjects
import rpy2.robjects.conversion
from .. import utils
from io import StringIO
from rpy2 import rinterface
from rpy2.robjects import r, vectors, globalenv
import rpy2.robjects.packages as rpacks
@pytest.mark.skipif(IPython is None, reason='The optional package IPython cannot be imported.')
@pytest.mark.skipif(not has_numpy, reason='numpy not installed')
def test_pull(ipython_with_magic, clean_globalenv):
    r('Z=c(11:20)')
    ipython_with_magic.run_line_magic('Rpull', 'Z')
    np.testing.assert_almost_equal(np.asarray(r('Z')), ipython_with_magic.user_ns['Z'])
    np.testing.assert_almost_equal(ipython_with_magic.user_ns['Z'], np.arange(11, 21))