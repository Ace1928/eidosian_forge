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
def JM_choice_options(annot):
    """
    return list of choices for list or combo boxes
    """
    annot_obj = mupdf.pdf_annot_obj(annot.this)
    if mupdf_version_tuple >= (1, 24):
        opts = mupdf.pdf_choice_widget_options2(annot, 0)
    else:

        def pdf_choice_widget_options(annot, exportval):
            optarr = mupdf.pdf_dict_get_inheritable(mupdf.pdf_annot_obj(annot.this), PDF_NAME('Opt'))
            n = mupdf.pdf_array_len(optarr)
            opts = []
            if not n:
                return opts
            optarr = mupdf.pdf_dict_get(annot_obj, PDF_NAME('Opt'))
            for i in range(n):
                m = mupdf.pdf_array_len(mupdf.pdf_array_get(optarr, i))
                if m == 2:
                    val = (mupdf.pdf_to_text_string(mupdf.pdf_array_get(mupdf.pdf_array_get(optarr, i), 0)), mupdf.pdf_to_text_string(mupdf.pdf_array_get(mupdf.pdf_array_get(optarr, i), 1)))
                    opts.append(val)
                else:
                    val = JM_UnicodeFromStr(mupdf.pdf_to_text_string(mupdf.pdf_array_get(optarr, i)))
                    opts.append(val)
            return opts
        opts = pdf_choice_widget_options(annot, 0)
    n = len(opts)
    if n == 0:
        return
    optarr = mupdf.pdf_dict_get(annot_obj, PDF_NAME('Opt'))
    liste = []
    for i in range(n):
        m = mupdf.pdf_array_len(mupdf.pdf_array_get(optarr, i))
        if m == 2:
            val = (mupdf.pdf_to_text_string(mupdf.pdf_array_get(mupdf.pdf_array_get(optarr, i), 0)), mupdf.pdf_to_text_string(mupdf.pdf_array_get(mupdf.pdf_array_get(optarr, i), 1)))
            liste.append(val)
        else:
            val = mupdf.pdf_to_text_string(mupdf.pdf_array_get(optarr, i))
            liste.append(val)
    return liste