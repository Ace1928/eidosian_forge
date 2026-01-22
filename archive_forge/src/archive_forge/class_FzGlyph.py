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
class FzGlyph(object):
    """
    Wrapper class for struct `fz_glyph`.
    Glyphs represent a run length encoded set of pixels for a 2
    dimensional region of a plane.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_glyph_bbox(self):
        """
        Class-aware wrapper for `::fz_glyph_bbox()`.
        	Return the bounding box of the glyph in pixels.
        """
        return _mupdf.FzGlyph_fz_glyph_bbox(self)

    def fz_glyph_bbox_no_ctx(self):
        """ Class-aware wrapper for `::fz_glyph_bbox_no_ctx()`."""
        return _mupdf.FzGlyph_fz_glyph_bbox_no_ctx(self)

    def fz_glyph_height(self):
        """
        Class-aware wrapper for `::fz_glyph_height()`.
        	Return the height of the glyph in pixels.
        """
        return _mupdf.FzGlyph_fz_glyph_height(self)

    def fz_glyph_width(self):
        """
        Class-aware wrapper for `::fz_glyph_width()`.
        	Return the width of the glyph in pixels.
        """
        return _mupdf.FzGlyph_fz_glyph_width(self)

    def __init__(self, *args):
        """
        *Overload 1:*
        Copy constructor using `fz_keep_glyph()`.

        |

        *Overload 2:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::fz_glyph`.
        """
        _mupdf.FzGlyph_swiginit(self, _mupdf.new_FzGlyph(*args))
    __swig_destroy__ = _mupdf.delete_FzGlyph

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzGlyph_m_internal_value(self)
    m_internal = property(_mupdf.FzGlyph_m_internal_get, _mupdf.FzGlyph_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzGlyph_s_num_instances_get, _mupdf.FzGlyph_s_num_instances_set)