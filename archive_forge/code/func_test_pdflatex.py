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
@needs_pgf_pdflatex
@pytest.mark.skipif(not _has_tex_package('type1ec'), reason='needs type1ec.sty')
@pytest.mark.skipif(not _has_tex_package('ucs'), reason='needs ucs.sty')
@pytest.mark.backend('pgf')
@image_comparison(['pgf_pdflatex.pdf'], style='default', tol=11.71 if _old_gs_version else 0)
def test_pdflatex():
    rc_pdflatex = {'font.family': 'serif', 'pgf.rcfonts': False, 'pgf.texsystem': 'pdflatex', 'pgf.preamble': '\\usepackage[utf8x]{inputenc}\\usepackage[T1]{fontenc}'}
    mpl.rcParams.update(rc_pdflatex)
    create_figure()