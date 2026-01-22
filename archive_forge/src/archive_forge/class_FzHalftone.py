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
class FzHalftone(object):
    """
    Wrapper class for struct `fz_halftone`.
    A halftone is a set of threshold tiles, one per component. Each
    threshold tile is a pixmap, possibly of varying sizes and
    phases. Currently, we only provide one 'default' halftone tile
    for operating on 1 component plus alpha pixmaps (where the alpha
    is ignored). This is signified by a fz_halftone pointer to NULL.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        Copy constructor using `fz_keep_halftone()`.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_halftone`.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::fz_halftone`.
        """
        _mupdf.FzHalftone_swiginit(self, _mupdf.new_FzHalftone(*args))
    __swig_destroy__ = _mupdf.delete_FzHalftone

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzHalftone_m_internal_value(self)
    m_internal = property(_mupdf.FzHalftone_m_internal_get, _mupdf.FzHalftone_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzHalftone_s_num_instances_get, _mupdf.FzHalftone_s_num_instances_set)