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
@pytest.mark.backend('pgf')
def test_sketch_params():
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    handle, = ax.plot([0, 1])
    handle.set_sketch_params(scale=5, length=30, randomness=42)
    with BytesIO() as fd:
        fig.savefig(fd, format='pgf')
        buf = fd.getvalue().decode()
    baseline = '\\pgfpathmoveto{\\pgfqpoint{0.375000in}{0.300000in}}%\n\\pgfpathlineto{\\pgfqpoint{2.700000in}{2.700000in}}%\n\\usepgfmodule{decorations}%\n\\usepgflibrary{decorations.pathmorphing}%\n\\pgfkeys{/pgf/decoration/.cd, segment length = 0.150000in, amplitude = 0.100000in}%\n\\pgfmathsetseed{42}%\n\\pgfdecoratecurrentpath{random steps}%\n\\pgfusepath{stroke}%'
    assert baseline in buf