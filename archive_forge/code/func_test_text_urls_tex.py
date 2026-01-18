import datetime
import decimal
import io
import os
from pathlib import Path
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import (
from matplotlib.cbook import _get_data_path
from matplotlib.ft2font import FT2Font
from matplotlib.font_manager import findfont, FontProperties
from matplotlib.backends._backend_pdf_ps import get_glyphs_subset
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
@needs_usetex
def test_text_urls_tex():
    pikepdf = pytest.importorskip('pikepdf')
    test_url = 'https://test_text_urls.matplotlib.org/'
    fig = plt.figure(figsize=(2, 1))
    fig.text(0.1, 0.7, 'test tex $123$', usetex=True, url=f'{test_url}tex')
    with io.BytesIO() as fd:
        fig.savefig(fd, format='pdf')
        with pikepdf.Pdf.open(fd) as pdf:
            annots = pdf.pages[0].Annots
            annot = next((a for a in annots if a.A.URI == f'{test_url}tex'), None)
            assert annot is not None
            assert annot.Rect[1] == decimal.Decimal('0.7') * 72