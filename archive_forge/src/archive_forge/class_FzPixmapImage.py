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
class FzPixmapImage(object):
    """ Wrapper class for struct `fz_pixmap_image`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_pixmap_image_tile(self):
        """
        Class-aware wrapper for `::fz_pixmap_image_tile()`.
        	Retrieve the underlying fz_pixmap for an image.

        	Returns a pointer to the underlying fz_pixmap for an image,
        	or NULL if this image is not based upon an fz_pixmap.

        	No reference is returned. Lifespan is limited to that of
        	the image itself. If required, use fz_keep_pixmap to take
        	a reference to keep it longer.
        """
        return _mupdf.FzPixmapImage_fz_pixmap_image_tile(self)

    def fz_set_pixmap_image_tile(self, pix):
        """ Class-aware wrapper for `::fz_set_pixmap_image_tile()`."""
        return _mupdf.FzPixmapImage_fz_set_pixmap_image_tile(self, pix)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_pixmap_image`.
        """
        _mupdf.FzPixmapImage_swiginit(self, _mupdf.new_FzPixmapImage(*args))
    __swig_destroy__ = _mupdf.delete_FzPixmapImage

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzPixmapImage_m_internal_value(self)
    m_internal = property(_mupdf.FzPixmapImage_m_internal_get, _mupdf.FzPixmapImage_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzPixmapImage_s_num_instances_get, _mupdf.FzPixmapImage_s_num_instances_set)