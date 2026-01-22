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
def _print_ps(self, fmt, outfile, *, metadata=None, papertype=None, orientation='portrait', bbox_inches_restore=None, **kwargs):
    dpi = self.figure.dpi
    self.figure.dpi = 72
    dsc_comments = {}
    if isinstance(outfile, (str, os.PathLike)):
        filename = pathlib.Path(outfile).name
        dsc_comments['Title'] = filename.encode('ascii', 'replace').decode('ascii')
    dsc_comments['Creator'] = (metadata or {}).get('Creator', f'Matplotlib v{mpl.__version__}, https://matplotlib.org/')
    source_date_epoch = os.getenv('SOURCE_DATE_EPOCH')
    dsc_comments['CreationDate'] = datetime.datetime.fromtimestamp(int(source_date_epoch), datetime.timezone.utc).strftime('%a %b %d %H:%M:%S %Y') if source_date_epoch else time.ctime()
    dsc_comments = '\n'.join((f'%%{k}: {v}' for k, v in dsc_comments.items()))
    if papertype is None:
        papertype = mpl.rcParams['ps.papersize']
    papertype = papertype.lower()
    _api.check_in_list(['figure', 'auto', *papersize], papertype=papertype)
    orientation = _api.check_getitem(_Orientation, orientation=orientation.lower())
    printer = self._print_figure_tex if mpl.rcParams['text.usetex'] else self._print_figure
    printer(fmt, outfile, dpi=dpi, dsc_comments=dsc_comments, orientation=orientation, papertype=papertype, bbox_inches_restore=bbox_inches_restore, **kwargs)