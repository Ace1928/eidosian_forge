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
def _make_PdfFilterOptions(recurse=0, instance_forms=0, ascii=0, no_update=0, sanitize=0, sopts=None):
    """
    Returns a mupdf.PdfFilterOptions instance.
    """
    filter_ = mupdf.PdfFilterOptions()
    filter_.recurse = recurse
    filter_.instance_forms = instance_forms
    filter_.ascii = ascii
    filter_.no_update = no_update
    if sanitize:
        if sopts:
            assert isinstance(sopts, mupdf.PdfSanitizeFilterOptions)
        else:
            sopts = mupdf.PdfSanitizeFilterOptions()

        class Factory(mupdf.PdfFilterFactory2):

            def __init__(self):
                super().__init__()
                self.use_virtual_filter()
                self.sopts = sopts

            def filter(self, ctx, doc, chain, struct_parents, transform, options):
                if 0:
                    log(f'sanitize filter.filter():')
                    log(f'    self={self!r}')
                    log(f'    ctx={ctx!r}')
                    log(f'    doc={doc!r}')
                    log(f'    chain={chain!r}')
                    log(f'    struct_parents={struct_parents!r}')
                    log(f'    transform={transform!r}')
                    log(f'    options={options!r}')
                    log(f'    self.sopts.internal()={self.sopts.internal()!r}')
                return mupdf.ll_pdf_new_sanitize_filter(doc, chain, struct_parents, transform, options, self.sopts.internal())
        factory = Factory()
        filter_.add_factory(factory.internal())
        filter_._factory = factory
    return filter_