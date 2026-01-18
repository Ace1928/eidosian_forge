import datetime
from io import BytesIO
import os
import shutil
import numpy as np
from packaging.version import parse as parse_version
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.testing import _has_tex_package, _check_for_pgf
from matplotlib.testing.exceptions import ImageComparisonFailure
from matplotlib.testing.compare import compare_images
from matplotlib.backends.backend_pgf import PdfPages
from matplotlib.testing.decorators import (
from matplotlib.testing._markers import (
@mpl.style.context('default')
@pytest.mark.backend('pgf')
@pytest.mark.parametrize('system', [pytest.param('lualatex', marks=[needs_pgf_lualatex]), pytest.param('pdflatex', marks=[needs_pgf_pdflatex]), pytest.param('xelatex', marks=[needs_pgf_xelatex])])
def test_pdf_pages(system):
    rc_pdflatex = {'font.family': 'serif', 'pgf.rcfonts': False, 'pgf.texsystem': system}
    mpl.rcParams.update(rc_pdflatex)
    fig1, ax1 = plt.subplots()
    ax1.plot(range(5))
    fig1.tight_layout()
    fig2, ax2 = plt.subplots(figsize=(3, 2))
    ax2.plot(range(5))
    fig2.tight_layout()
    path = os.path.join(result_dir, f'pdfpages_{system}.pdf')
    md = {'Author': 'me', 'Title': 'Multipage PDF with pgf', 'Subject': 'Test page', 'Keywords': 'test,pdf,multipage', 'ModDate': datetime.datetime(1968, 8, 1, tzinfo=datetime.timezone(datetime.timedelta(0))), 'Trapped': 'Unknown'}
    with PdfPages(path, metadata=md) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig1)
        assert pdf.get_pagecount() == 3