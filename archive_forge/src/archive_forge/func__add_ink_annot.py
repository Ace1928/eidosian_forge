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
def _add_ink_annot(self, list):
    page = mupdf.pdf_page_from_fz_page(self.this)
    ASSERT_PDF(page)
    if not PySequence_Check(list):
        raise ValueError(MSG_BAD_ARG_INK_ANNOT)
    ctm = mupdf.FzMatrix()
    mupdf.pdf_page_transform(page, mupdf.FzRect(0), ctm)
    inv_ctm = mupdf.fz_invert_matrix(ctm)
    annot = mupdf.pdf_create_annot(page, mupdf.PDF_ANNOT_INK)
    annot_obj = mupdf.pdf_annot_obj(annot)
    n0 = len(list)
    inklist = mupdf.pdf_new_array(page.doc(), n0)
    for j in range(n0):
        sublist = list[j]
        n1 = len(sublist)
        stroke = mupdf.pdf_new_array(page.doc(), 2 * n1)
        for i in range(n1):
            p = sublist[i]
            if not PySequence_Check(p) or PySequence_Size(p) != 2:
                raise ValueError(MSG_BAD_ARG_INK_ANNOT)
            point = mupdf.fz_transform_point(JM_point_from_py(p), inv_ctm)
            mupdf.pdf_array_push_real(stroke, point.x)
            mupdf.pdf_array_push_real(stroke, point.y)
        mupdf.pdf_array_push(inklist, stroke)
    mupdf.pdf_dict_put(annot_obj, PDF_NAME('InkList'), inklist)
    mupdf.pdf_update_annot(annot)
    JM_add_annot_id(annot, 'A')
    return Annot(annot)