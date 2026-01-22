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
class JM_image_reporter_Filter(mupdf.PdfFilterOptions2):

    def __init__(self):
        super().__init__()
        self.use_virtual_image_filter()

    def image_filter(self, ctx, ctm, name, image):
        assert isinstance(ctm, mupdf.fz_matrix)
        JM_image_filter(self, mupdf.FzMatrix(ctm), name, image)
        if mupdf_cppyy:
            return 0