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
def add_freetext_annot(self, rect: rect_like, text: str, fontsize: float=11, fontname: OptStr=None, border_color: OptSeq=None, text_color: OptSeq=None, fill_color: OptSeq=None, align: int=0, rotate: int=0) -> Annot:
    """Add a 'FreeText' annotation."""
    old_rotation = annot_preprocess(self)
    try:
        annot = self._add_freetext_annot(rect, text, fontsize=fontsize, fontname=fontname, border_color=border_color, text_color=text_color, fill_color=fill_color, align=align, rotate=rotate)
    finally:
        if old_rotation != 0:
            self.set_rotation(old_rotation)
    annot_postprocess(self, annot)
    return annot