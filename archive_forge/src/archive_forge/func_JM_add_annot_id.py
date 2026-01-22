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
def JM_add_annot_id(annot, stem):
    """
    Add a unique /NM key to an annotation or widget.
    Append a number to 'stem' such that the result is a unique name.
    """
    assert isinstance(annot, mupdf.PdfAnnot)
    page = mupdf.pdf_annot_page(annot)
    annot_obj = mupdf.pdf_annot_obj(annot)
    names = JM_get_annot_id_list(page)
    i = 0
    while 1:
        stem_id = f'{JM_annot_id_stem}-{stem}{i}'
        if stem_id not in names:
            break
        i += 1
    response = JM_StrAsChar(stem_id)
    name = mupdf.pdf_new_string(response, len(response))
    mupdf.pdf_dict_puts(annot_obj, 'NM', name)
    page.doc().m_internal.resynth_required = 0