from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class fz_stext_block(object):
    """
    A text block is a list of lines of text (typically a paragraph),
    or an image.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    type = property(_mupdf.fz_stext_block_type_get, _mupdf.fz_stext_block_type_set)
    bbox = property(_mupdf.fz_stext_block_bbox_get, _mupdf.fz_stext_block_bbox_set)
    prev = property(_mupdf.fz_stext_block_prev_get, _mupdf.fz_stext_block_prev_set)
    next = property(_mupdf.fz_stext_block_next_get, _mupdf.fz_stext_block_next_set)

    def __init__(self):
        _mupdf.fz_stext_block_swiginit(self, _mupdf.new_fz_stext_block())
    __swig_destroy__ = _mupdf.delete_fz_stext_block