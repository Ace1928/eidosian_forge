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
class JM_new_texttrace_device(mupdf.FzDevice2):
    """
    Trace TEXT device for Python method Page.get_texttrace()
    """

    def __init__(self, out):
        super().__init__()
        self.use_virtual_fill_path()
        self.use_virtual_stroke_path()
        self.use_virtual_fill_text()
        self.use_virtual_stroke_text()
        self.use_virtual_ignore_text()
        self.use_virtual_fill_shade()
        self.use_virtual_fill_image()
        self.use_virtual_fill_image_mask()
        self.use_virtual_begin_layer()
        self.use_virtual_end_layer()
        self.out = out
        self.seqno = 0
        self.depth = 0
        self.clips = 0
        self.method = None
        self.seqno = 0
        self.pathdict = dict()
        self.scissors = list()
        self.linewidth = 0
        self.ptm = mupdf.FzMatrix()
        self.ctm = mupdf.FzMatrix()
        self.rot = mupdf.FzMatrix()
        self.lastpoint = mupdf.FzPoint()
        self.pathrect = mupdf.FzRect()
        self.pathfactor = 0
        self.linecount = 0
        self.path_type = 0
        self.layer_name = ''
    fill_path = jm_increase_seqno
    stroke_path = jm_dev_linewidth
    fill_text = jm_lineart_fill_text
    stroke_text = jm_lineart_stroke_text
    ignore_text = jm_lineart_ignore_text
    fill_shade = jm_increase_seqno
    fill_image = jm_increase_seqno
    fill_image_mask = jm_increase_seqno
    begin_layer = jm_lineart_begin_layer
    end_layer = jm_lineart_end_layer