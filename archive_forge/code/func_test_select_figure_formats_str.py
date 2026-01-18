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
def test_select_figure_formats_str():
    ip = get_ipython()
    for fmt, active_mime in _fmt_mime_map.items():
        pt.select_figure_formats(ip, fmt)
        for mime, f in ip.display_formatter.formatters.items():
            if mime == active_mime:
                assert Figure in f
            else:
                assert Figure not in f