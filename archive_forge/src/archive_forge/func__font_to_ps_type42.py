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
def _font_to_ps_type42(font_path, chars, fh):
    """
    Subset *chars* from the font at *font_path* into a Type 42 font at *fh*.

    Parameters
    ----------
    font_path : path-like
        Path to the font to be subsetted.
    chars : str
        The characters to include in the subsetted font.
    fh : file-like
        Where to write the font.
    """
    subset_str = ''.join((chr(c) for c in chars))
    _log.debug('SUBSET %s characters: %s', font_path, subset_str)
    try:
        fontdata = _backend_pdf_ps.get_glyphs_subset(font_path, subset_str)
        _log.debug('SUBSET %s %d -> %d', font_path, os.stat(font_path).st_size, fontdata.getbuffer().nbytes)
        font = FT2Font(fontdata)
        glyph_ids = [font.get_char_index(c) for c in chars]
        with TemporaryDirectory() as tmpdir:
            tmpfile = os.path.join(tmpdir, 'tmp.ttf')
            with open(tmpfile, 'wb') as tmp:
                tmp.write(fontdata.getvalue())
            convert_ttf_to_ps(os.fsencode(tmpfile), fh, 42, glyph_ids)
    except RuntimeError:
        _log.warning('The PostScript backend does not currently support the selected font.')
        raise