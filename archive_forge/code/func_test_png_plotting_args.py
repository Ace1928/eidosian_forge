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
def test_png_plotting_args(ipython_with_magic, clean_globalenv):
    """Exercise the PNG plotting machinery"""
    ipython_with_magic.push({'x': np.arange(5), 'y': np.array([3, 5, 4, 6, 7])})
    cell = "\n    plot(x, y, pch=23, bg='orange', cex=2)\n    "
    png_px_args = [' '.join(('--input=x,y --units=px', w, h, p)) for w, h, p in product(['--width=400 ', ''], ['--height=400', ''], ['-p=10', ''])]
    for line in png_px_args:
        ipython_with_magic.run_line_magic('Rdevice', 'png')
        ipython_with_magic.run_cell_magic('R', line, cell)