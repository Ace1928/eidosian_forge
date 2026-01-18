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
def test_push_localscope(ipython_with_magic, clean_globalenv):
    """Test that Rpush looks for variables in the local scope first."""
    ipython_with_magic.run_cell(textwrap.dedent('\n            def rmagic_addone(u):\n                %Rpush u\n                %R result = u+1\n                %Rpull result\n                return result[0]\n            u = 0\n            result = rmagic_addone(12344)\n            '))
    result = ipython_with_magic.user_ns['result']
    np.testing.assert_equal(result, 12345)