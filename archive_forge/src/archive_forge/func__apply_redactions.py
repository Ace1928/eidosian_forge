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
def _apply_redactions(self, text, images, graphics):
    page = self._pdf_page()
    opts = mupdf.PdfRedactOptions()
    opts.black_boxes = 0
    opts.text = text
    opts.image_method = images
    opts.line_art = graphics
    ASSERT_PDF(page)
    success = mupdf.pdf_redact_page(page.doc(), page, opts)
    return success