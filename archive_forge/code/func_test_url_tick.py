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
def test_url_tick(monkeypatch):
    monkeypatch.setenv('SOURCE_DATE_EPOCH', '19680801')
    fig1, ax = plt.subplots()
    ax.scatter([1, 2, 3], [4, 5, 6])
    for i, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.set_url(f'https://example.com/{i}')
    fig2, ax = plt.subplots()
    ax.scatter([1, 2, 3], [4, 5, 6])
    for i, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.label1.set_url(f'https://example.com/{i}')
        tick.label2.set_url(f'https://example.com/{i}')
    b1 = BytesIO()
    fig1.savefig(b1, format='svg')
    b1 = b1.getvalue()
    b2 = BytesIO()
    fig2.savefig(b2, format='svg')
    b2 = b2.getvalue()
    for i in range(len(ax.yaxis.get_major_ticks())):
        assert f'https://example.com/{i}'.encode('ascii') in b1
    assert b1 == b2