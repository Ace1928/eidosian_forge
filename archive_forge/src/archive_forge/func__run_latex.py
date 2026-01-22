import codecs
import datetime
import functools
from io import BytesIO
import logging
import math
import os
import pathlib
import shutil
import subprocess
from tempfile import TemporaryDirectory
import weakref
from PIL import Image
import matplotlib as mpl
from matplotlib import _api, cbook, font_manager as fm
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.backends.backend_pdf import (
from matplotlib.path import Path
from matplotlib.figure import Figure
from matplotlib._pylab_helpers import Gcf
def _run_latex(self):
    texcommand = mpl.rcParams['pgf.texsystem']
    with TemporaryDirectory() as tmpdir:
        tex_source = pathlib.Path(tmpdir, 'pdf_pages.tex')
        tex_source.write_bytes(self._file.getvalue())
        cbook._check_and_log_subprocess([texcommand, '-interaction=nonstopmode', '-halt-on-error', tex_source], _log, cwd=tmpdir)
        shutil.move(tex_source.with_suffix('.pdf'), self._output_name)