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
class fz_layout_block(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    pool = property(_mupdf.fz_layout_block_pool_get, _mupdf.fz_layout_block_pool_set)
    matrix = property(_mupdf.fz_layout_block_matrix_get, _mupdf.fz_layout_block_matrix_set)
    inv_matrix = property(_mupdf.fz_layout_block_inv_matrix_get, _mupdf.fz_layout_block_inv_matrix_set)
    head = property(_mupdf.fz_layout_block_head_get, _mupdf.fz_layout_block_head_set)
    tailp = property(_mupdf.fz_layout_block_tailp_get, _mupdf.fz_layout_block_tailp_set)
    text_tailp = property(_mupdf.fz_layout_block_text_tailp_get, _mupdf.fz_layout_block_text_tailp_set)

    def __init__(self):
        _mupdf.fz_layout_block_swiginit(self, _mupdf.new_fz_layout_block())
    __swig_destroy__ = _mupdf.delete_fz_layout_block