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
def _addAnnot_FromString(self, linklist):
    """Add links from list of object sources."""
    CheckParent(self)
    if g_use_extra:
        self.__class__._addAnnot_FromString = extra.Page_addAnnot_FromString
        return extra.Page_addAnnot_FromString(self.this, linklist)
    page = mupdf.pdf_page_from_fz_page(self.this)
    lcount = len(linklist)
    if lcount < 1:
        return
    i = -1
    ASSERT_PDF(page)
    if not isinstance(linklist, tuple):
        raise ValueError("bad 'linklist' argument")
    if not mupdf.pdf_dict_get(page.obj(), PDF_NAME('Annots')).m_internal:
        mupdf.pdf_dict_put_array(page.obj(), PDF_NAME('Annots'), lcount)
    annots = mupdf.pdf_dict_get(page.obj(), PDF_NAME('Annots'))
    assert annots.m_internal, f'lcount={lcount!r} annots.m_internal={annots.m_internal!r}'
    for i in range(lcount):
        txtpy = linklist[i]
        text = JM_StrAsChar(txtpy)
        if not text:
            PySys_WriteStderr('skipping bad link / annot item %i.\n', i)
            continue
        try:
            annot = mupdf.pdf_add_object(page.doc(), JM_pdf_obj_from_str(page.doc(), text))
            ind_obj = mupdf.pdf_new_indirect(page.doc(), mupdf.pdf_to_num(annot), 0)
            mupdf.pdf_array_push(annots, ind_obj)
        except Exception:
            if g_exceptions_verbose:
                exception_info()
            message('skipping bad link / annot item %i.\n' % i, file=sys.stderr)