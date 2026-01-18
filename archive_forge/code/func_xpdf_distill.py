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
def xpdf_distill(tmpfile, eps=False, ptype='letter', bbox=None, rotated=False):
    """
    Use ghostscript's ps2pdf and xpdf's/poppler's pdftops to distill a file.
    This yields smaller files without illegal encapsulated postscript
    operators. This distiller is preferred, generating high-level postscript
    output that treats text as text.
    """
    mpl._get_executable_info('gs')
    mpl._get_executable_info('pdftops')
    if eps:
        paper_option = ['-dEPSCrop']
    elif ptype == 'figure':
        paper_option = [f'-dDEVICEWIDTHPOINTS#{bbox[2]}', f'-dDEVICEHEIGHTPOINTS#{bbox[3]}']
    else:
        paper_option = [f'-sPAPERSIZE#{ptype}']
    with TemporaryDirectory() as tmpdir:
        tmppdf = pathlib.Path(tmpdir, 'tmp.pdf')
        tmpps = pathlib.Path(tmpdir, 'tmp.ps')
        cbook._check_and_log_subprocess(['ps2pdf', '-dAutoFilterColorImages#false', '-dAutoFilterGrayImages#false', '-sAutoRotatePages#None', '-sGrayImageFilter#FlateEncode', '-sColorImageFilter#FlateEncode', *paper_option, tmpfile, tmppdf], _log)
        cbook._check_and_log_subprocess(['pdftops', '-paper', 'match', '-level3', tmppdf, tmpps], _log)
        shutil.move(tmpps, tmpfile)
    if eps:
        pstoeps(tmpfile)