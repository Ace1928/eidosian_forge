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