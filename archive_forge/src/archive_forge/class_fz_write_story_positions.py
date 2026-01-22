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
class fz_write_story_positions(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    positions = property(_mupdf.fz_write_story_positions_positions_get, _mupdf.fz_write_story_positions_positions_set)
    num = property(_mupdf.fz_write_story_positions_num_get, _mupdf.fz_write_story_positions_num_set)

    def __init__(self):
        _mupdf.fz_write_story_positions_swiginit(self, _mupdf.new_fz_write_story_positions())
    __swig_destroy__ = _mupdf.delete_fz_write_story_positions