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
def _load_annot(self, name, xref):
    page = self._pdf_page()
    ASSERT_PDF(page)
    if xref == 0:
        annot = JM_get_annot_by_name(page, name)
    else:
        annot = JM_get_annot_by_xref(page, xref)
    return Annot(annot) if annot else None