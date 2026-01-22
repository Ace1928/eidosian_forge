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
def JM_get_ocg_arrays(conf):
    rc = dict()
    arr = mupdf.pdf_dict_get(conf, PDF_NAME('ON'))
    list_ = JM_get_ocg_arrays_imp(arr)
    if list_:
        rc['on'] = list_
    arr = mupdf.pdf_dict_get(conf, PDF_NAME('OFF'))
    list_ = JM_get_ocg_arrays_imp(arr)
    if list_:
        rc['off'] = list_
    arr = mupdf.pdf_dict_get(conf, PDF_NAME('Locked'))
    list_ = JM_get_ocg_arrays_imp(arr)
    if list_:
        rc['locked'] = list_
    list_ = list()
    arr = mupdf.pdf_dict_get(conf, PDF_NAME('RBGroups'))
    if mupdf.pdf_is_array(arr):
        n = mupdf.pdf_array_len(arr)
        for i in range(n):
            obj = mupdf.pdf_array_get(arr, i)
            list1 = JM_get_ocg_arrays_imp(obj)
            list_.append(list1)
    if list_:
        rc['rbgroups'] = list_
    obj = mupdf.pdf_dict_get(conf, PDF_NAME('BaseState'))
    if obj.m_internal:
        state = mupdf.pdf_to_name(obj)
        rc['basestate'] = state
    return rc