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
def add_polygon_annot(self, points: list) -> Annot:
    """Add a 'Polygon' annotation."""
    old_rotation = annot_preprocess(self)
    try:
        annot = self._add_multiline(points, mupdf.PDF_ANNOT_POLYGON)
    finally:
        if old_rotation != 0:
            self.set_rotation(old_rotation)
    annot_postprocess(self, annot)
    return annot