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
def xref_xml_metadata(self):
    """Get xref of document XML metadata."""
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    root = mupdf.pdf_dict_get(mupdf.pdf_trailer(pdf), PDF_NAME('Root'))
    if not root.m_internal:
        RAISEPY(MSG_BAD_PDFROOT, JM_Exc_FileDataError)
    xml = mupdf.pdf_dict_get(root, PDF_NAME('Metadata'))
    xref = 0
    if xml.m_internal:
        xref = mupdf.pdf_to_num(xml)
    return xref