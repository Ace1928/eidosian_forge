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
@pytest.mark.parametrize('rcode,exception_expr', (('"a" + 1', 'rmagic.RInterpreterError'), ('"a" + ', 'rpy2.rinterface_lib._rinterface_capi.RParsingError')))
def test_run_cell_with_error(ipython_with_magic, clean_globalenv, rcode, exception_expr):
    """Run an R block with an error."""
    with pytest.raises(eval(exception_expr)):
        ipython_with_magic.run_line_magic('R', rcode)