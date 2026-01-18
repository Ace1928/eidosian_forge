import datetime
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree
import xml.parsers.expat
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.text import Text
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
from matplotlib import font_manager as fm
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
def test_svg_default_metadata(monkeypatch):
    monkeypatch.setenv('SOURCE_DATE_EPOCH', '19680801')
    fig, ax = plt.subplots()
    with BytesIO() as fd:
        fig.savefig(fd, format='svg')
        buf = fd.getvalue().decode()
    assert mpl.__version__ in buf
    assert '1970-08-16' in buf
    assert 'image/svg+xml' in buf
    assert 'StillImage' in buf
    with BytesIO() as fd:
        fig.savefig(fd, format='svg', metadata={'Date': None, 'Creator': None, 'Format': None, 'Type': None})
        buf = fd.getvalue().decode()
    assert mpl.__version__ not in buf
    assert '1970-08-16' not in buf
    assert 'image/svg+xml' not in buf
    assert 'StillImage' not in buf