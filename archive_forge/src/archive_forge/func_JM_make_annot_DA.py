import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def JM_make_annot_DA(annot, ncol, col, fontname, fontsize):
    buf = ''
    if ncol < 1:
        buf += f'0 g '
    elif ncol == 1:
        buf += f'{col[0]:g} g '
    elif ncol == 2:
        assert 0
    elif ncol == 3:
        buf += f'{col[0]:g} {col[1]:g} {col[2]:g} rg '
    else:
        buf += f'{col[0]:g} {col[1]:g} {col[2]:g} {col[3]:g} k '
    buf += f'/{JM_expand_fname(fontname)} {fontsize} Tf'
    mupdf.pdf_dict_put_text_string(mupdf.pdf_annot_obj(annot), mupdf.PDF_ENUM_NAME_DA, buf)