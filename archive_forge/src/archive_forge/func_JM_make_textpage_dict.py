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
def JM_make_textpage_dict(tp, page_dict, raw):
    if g_use_extra:
        return extra.JM_make_textpage_dict(tp.m_internal, page_dict, raw)
    text_buffer = mupdf.fz_new_buffer(128)
    block_list = []
    tp_rect = mupdf.FzRect(tp.m_internal.mediabox)
    block_n = -1
    for block in tp:
        block_n += 1
        if not mupdf.fz_contains_rect(tp_rect, mupdf.FzRect(block.m_internal.bbox)) and (not mupdf.fz_is_infinite_rect(tp_rect)) and (block.m_internal.type == mupdf.FZ_STEXT_BLOCK_IMAGE):
            continue
        if not mupdf.fz_is_infinite_rect(tp_rect) and mupdf.fz_is_empty_rect(mupdf.fz_intersect_rect(tp_rect, mupdf.FzRect(block.m_internal.bbox))):
            continue
        block_dict = dict()
        block_dict[dictkey_number] = block_n
        block_dict[dictkey_type] = block.m_internal.type
        if block.m_internal.type == mupdf.FZ_STEXT_BLOCK_IMAGE:
            block_dict[dictkey_bbox] = JM_py_from_rect(block.m_internal.bbox)
            JM_make_image_block(block, block_dict)
        else:
            JM_make_text_block(block, block_dict, raw, text_buffer, tp_rect)
        block_list.append(block_dict)
    page_dict[dictkey_blocks] = block_list