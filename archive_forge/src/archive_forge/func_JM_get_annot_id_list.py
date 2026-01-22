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
def JM_get_annot_id_list(page):
    names = []
    annots = mupdf.pdf_dict_get(page.obj(), mupdf.PDF_ENUM_NAME_Annots)
    if not annots.m_internal:
        return names
    for i in range(mupdf.pdf_array_len(annots)):
        annot_obj = mupdf.pdf_array_get(annots, i)
        name = mupdf.pdf_dict_gets(annot_obj, 'NM')
        if name.m_internal:
            names.append(mupdf.pdf_to_text_string(name))
    return names