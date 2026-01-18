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
def test_svgnone_with_data_coordinates():
    plt.rcParams.update({'svg.fonttype': 'none', 'font.stretch': 'condensed'})
    expected = 'Unlikely to appear by chance'
    fig, ax = plt.subplots()
    ax.text(np.datetime64('2019-06-30'), 1, expected)
    ax.set_xlim(np.datetime64('2019-01-01'), np.datetime64('2019-12-31'))
    ax.set_ylim(0, 2)
    with BytesIO() as fd:
        fig.savefig(fd, format='svg')
        fd.seek(0)
        buf = fd.read().decode()
    assert expected in buf and 'condensed' in buf