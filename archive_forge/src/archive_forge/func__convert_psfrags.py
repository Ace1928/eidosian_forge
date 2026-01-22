import codecs
import datetime
from enum import Enum
import functools
from io import StringIO
import itertools
import logging
import os
import pathlib
import shutil
from tempfile import TemporaryDirectory
import time
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _path, _text_helpers
from matplotlib._afm import AFM
from matplotlib.backend_bases import (
from matplotlib.cbook import is_writable_file_like, file_requires_unicode
from matplotlib.font_manager import get_font
from matplotlib.ft2font import LOAD_NO_SCALE, FT2Font
from matplotlib._ttconv import convert_ttf_to_ps
from matplotlib._mathtext_data import uni2type1
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_mixed import MixedModeRenderer
from . import _backend_pdf_ps
def _convert_psfrags(tmppath, psfrags, paper_width, paper_height, orientation):
    """
    When we want to use the LaTeX backend with postscript, we write PSFrag tags
    to a temporary postscript file, each one marking a position for LaTeX to
    render some text. convert_psfrags generates a LaTeX document containing the
    commands to convert those tags to text. LaTeX/dvips produces the postscript
    file that includes the actual text.
    """
    with mpl.rc_context({'text.latex.preamble': mpl.rcParams['text.latex.preamble'] + mpl.texmanager._usepackage_if_not_loaded('color') + mpl.texmanager._usepackage_if_not_loaded('graphicx') + mpl.texmanager._usepackage_if_not_loaded('psfrag') + '\\geometry{papersize={%(width)sin,%(height)sin},margin=0in}' % {'width': paper_width, 'height': paper_height}}):
        dvifile = TexManager().make_dvi('\n\\begin{figure}\n  \\centering\\leavevmode\n  %(psfrags)s\n  \\includegraphics*[angle=%(angle)s]{%(epsfile)s}\n\\end{figure}' % {'psfrags': '\n'.join(psfrags), 'angle': 90 if orientation == 'landscape' else 0, 'epsfile': tmppath.resolve().as_posix()}, fontsize=10)
    with TemporaryDirectory() as tmpdir:
        psfile = os.path.join(tmpdir, 'tmp.ps')
        cbook._check_and_log_subprocess(['dvips', '-q', '-R0', '-o', psfile, dvifile], _log)
        shutil.move(psfile, tmppath)
    with open(tmppath) as fh:
        psfrag_rotated = 'Landscape' in fh.read(1000)
    return psfrag_rotated