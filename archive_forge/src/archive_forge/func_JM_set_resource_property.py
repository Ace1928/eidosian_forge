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
def JM_set_resource_property(ref, name, xref):
    """
    Insert an item into Resources/Properties (used for Marked Content)
    Arguments:
    (1) e.g. page object, Form XObject
    (2) marked content name
    (3) xref of the referenced object (insert as indirect reference)
    """
    pdf = mupdf.pdf_get_bound_document(ref)
    ind = mupdf.pdf_new_indirect(pdf, xref, 0)
    if not ind.m_internal:
        RAISEPY(MSG_BAD_XREF, PyExc_ValueError)
    resources = mupdf.pdf_dict_get(ref, PDF_NAME('Resources'))
    if not resources.m_internal:
        resources = mupdf.pdf_dict_put_dict(ref, PDF_NAME('Resources'), 1)
    properties = mupdf.pdf_dict_get(resources, PDF_NAME('Properties'))
    if not properties.m_internal:
        properties = mupdf.pdf_dict_put_dict(resources, PDF_NAME('Properties'), 1)
    mupdf.pdf_dict_put(properties, mupdf.pdf_new_name(name), ind)