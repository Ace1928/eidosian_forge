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
def JM_image_reporter(page):
    doc = page.doc()
    global g_img_info_matrix
    g_img_info_matrix = mupdf.FzMatrix()
    mediabox = mupdf.FzRect()
    mupdf.pdf_page_transform(page, mediabox, g_img_info_matrix)

    class SanitizeFilterOptions(mupdf.PdfSanitizeFilterOptions2):

        def __init__(self):
            super().__init__()
            self.use_virtual_image_filter()
        if mupdf_version_tuple >= (1, 23, 11):

            def image_filter(self, ctx, ctm, name, image, scissor):
                JM_image_filter(None, mupdf.FzMatrix(ctm), name, image)
        else:

            def image_filter(self, ctx, ctm, name, image):
                JM_image_filter(None, mupdf.FzMatrix(ctm), name, image)
    sanitize_filter_options = SanitizeFilterOptions()
    filter_options = _make_PdfFilterOptions(instance_forms=1, ascii=1, no_update=1, sanitize=1, sopts=sanitize_filter_options)
    global g_img_info
    g_img_info = []
    mupdf.pdf_filter_page_contents(doc, page, filter_options)
    rc = tuple(g_img_info)
    g_img_info = []
    return rc