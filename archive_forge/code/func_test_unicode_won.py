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
@mpl.style.context('default')
@needs_usetex
def test_unicode_won():
    fig = Figure()
    fig.text(0.5, 0.5, '\\textwon', usetex=True)
    with BytesIO() as fd:
        fig.savefig(fd, format='svg')
        buf = fd.getvalue()
    tree = xml.etree.ElementTree.fromstring(buf)
    ns = 'http://www.w3.org/2000/svg'
    won_id = 'SFSS3583-8e'
    assert len(tree.findall(f'.//{{{ns}}}path[@d][@id="{won_id}"]')) == 1
    assert f'#{won_id}' in tree.find(f'.//{{{ns}}}use').attrib.values()