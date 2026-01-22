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
def JM_merge_range(doc_des, doc_src, spage, epage, apage, rotate, links, annots, show_progress, graft_map):
    """
    Copy a range of pages (spage, epage) from a source PDF to a specified
    location (apage) of the target PDF.
    If spage > epage, the sequence of source pages is reversed.
    """
    if g_use_extra:
        return extra.JM_merge_range(doc_des, doc_src, spage, epage, apage, rotate, links, annots, show_progress, graft_map)
    afterpage = apage
    counter = 0
    total = mupdf.fz_absi(epage - spage) + 1
    if spage < epage:
        page = spage
        while page <= epage:
            page_merge(doc_des, doc_src, page, afterpage, rotate, links, annots, graft_map)
            counter += 1
            if show_progress > 0 and counter % show_progress == 0:
                sys.stdout.write('Inserted %i of %i pages.\n', counter, total)
            page += 1
            afterpage += 1
    else:
        page = spage
        while page >= epage:
            page_merge(doc_des, doc_src, page, afterpage, rotate, links, annots, graft_map)
            counter += 1
            if show_progress > 0 and counter % show_progress == 0:
                sys.stdout.write('Inserted %i of %i pages.\n', counter, total)
            page -= 1
            afterpage += 1