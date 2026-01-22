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
class Colorspace:

    def __init__(self, type_):
        """Supported are GRAY, RGB and CMYK."""
        if isinstance(type_, mupdf.FzColorspace):
            self.this = type_
        elif type_ == CS_GRAY:
            self.this = mupdf.FzColorspace(mupdf.FzColorspace.Fixed_GRAY)
        elif type_ == CS_CMYK:
            self.this = mupdf.FzColorspace(mupdf.FzColorspace.Fixed_CMYK)
        elif type_ == CS_RGB:
            self.this = mupdf.FzColorspace(mupdf.FzColorspace.Fixed_RGB)
        else:
            self.this = mupdf.FzColorspace(mupdf.FzColorspace.Fixed_RGB)

    def __repr__(self):
        x = ('', 'GRAY', '', 'RGB', 'CMYK')[self.n]
        return 'Colorspace(CS_%s) - %s' % (x, self.name)

    def _name(self):
        return mupdf.fz_colorspace_name(self.this)

    @property
    def n(self):
        """Size of one pixel."""
        return mupdf.fz_colorspace_n(self.this)

    @property
    def name(self):
        """Name of the Colorspace."""
        return self._name()