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
def test_pdf_pages_metadata_check(monkeypatch, system):
    pikepdf = pytest.importorskip('pikepdf')
    monkeypatch.setenv('SOURCE_DATE_EPOCH', '0')
    mpl.rcParams.update({'pgf.texsystem': system})
    fig, ax = plt.subplots()
    ax.plot(range(5))
    md = {'Author': 'me', 'Title': 'Multipage PDF with pgf', 'Subject': 'Test page', 'Keywords': 'test,pdf,multipage', 'ModDate': datetime.datetime(1968, 8, 1, tzinfo=datetime.timezone(datetime.timedelta(0))), 'Trapped': 'True'}
    path = os.path.join(result_dir, f'pdfpages_meta_check_{system}.pdf')
    with PdfPages(path, metadata=md) as pdf:
        pdf.savefig(fig)
    with pikepdf.Pdf.open(path) as pdf:
        info = {k: str(v) for k, v in pdf.docinfo.items()}
    if '/PTEX.FullBanner' in info:
        del info['/PTEX.FullBanner']
    if '/PTEX.Fullbanner' in info:
        del info['/PTEX.Fullbanner']
    producer = info.pop('/Producer')
    assert producer == f'Matplotlib pgf backend v{mpl.__version__}' or (system == 'lualatex' and 'LuaTeX' in producer)
    assert info == {'/Author': 'me', '/CreationDate': 'D:19700101000000Z', '/Creator': f'Matplotlib v{mpl.__version__}, https://matplotlib.org', '/Keywords': 'test,pdf,multipage', '/ModDate': 'D:19680801000000Z', '/Subject': 'Test page', '/Title': 'Multipage PDF with pgf', '/Trapped': '/True'}