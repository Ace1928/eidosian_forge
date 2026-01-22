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
def JM_get_annot_xref_list2(page):
    page = page._pdf_page()
    if not page.m_internal:
        return list()
    return JM_get_annot_xref_list(page.obj())