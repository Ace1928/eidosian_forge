from binascii import a2b_base64
from io import BytesIO
import pytest
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import numpy as np
from IPython.core.getipython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.display import _PNG, _JPEG
from .. import pylabtools as pt
from IPython.testing import decorators as dec
def test_qt(self):
    s = self.Shell()
    gui, backend = s.enable_matplotlib(None)
    assert gui == 'qt'
    assert s.pylab_gui_select == 'qt'
    gui, backend = s.enable_matplotlib('inline')
    assert gui == 'inline'
    assert s.pylab_gui_select == 'qt'
    gui, backend = s.enable_matplotlib('qt')
    assert gui == 'qt'
    assert s.pylab_gui_select == 'qt'
    gui, backend = s.enable_matplotlib('inline')
    assert gui == 'inline'
    assert s.pylab_gui_select == 'qt'
    gui, backend = s.enable_matplotlib()
    assert gui == 'qt'
    assert s.pylab_gui_select == 'qt'