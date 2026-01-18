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
@pytest.mark.skipif(not rpacks.isinstalled('Cairo'), reason='R package "Cairo" not installed')
def test_svg_plotting_args(ipython_with_magic, clean_globalenv):
    """Exercise the plotting machinery

    To pass SVG tests, we need Cairo installed in R."""
    ipython_with_magic.push({'x': np.arange(5), 'y': np.array([3, 5, 4, 6, 7])})
    cell = textwrap.dedent("\n    plot(x, y, pch=23, bg='orange', cex=2)\n    ")
    basic_args = [' '.join((w, h, p)) for w, h, p in product(['--width=6 ', ''], ['--height=6', ''], ['-p=10', ''])]
    for line in basic_args:
        ipython_with_magic.run_line_magic('Rdevice', 'svg')
        ipython_with_magic.run_cell_magic('R', line, cell)
    png_args = ['--units=in --res=1 ' + s for s in basic_args]
    for line in png_args:
        ipython_with_magic.run_line_magic('Rdevice', 'png')
        ipython_with_magic.run_cell_magic('R', line, cell)