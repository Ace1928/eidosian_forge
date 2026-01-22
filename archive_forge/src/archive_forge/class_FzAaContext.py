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
class FzAaContext(object):
    """ Wrapper class for struct `fz_aa_context`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        We use default copy constructor and operator=.  Default constructor, sets each member to default value.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_aa_context`.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::fz_aa_context`.
        """
        _mupdf.FzAaContext_swiginit(self, _mupdf.new_FzAaContext(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.FzAaContext_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_FzAaContext
    hscale = property(_mupdf.FzAaContext_hscale_get, _mupdf.FzAaContext_hscale_set)
    vscale = property(_mupdf.FzAaContext_vscale_get, _mupdf.FzAaContext_vscale_set)
    scale = property(_mupdf.FzAaContext_scale_get, _mupdf.FzAaContext_scale_set)
    bits = property(_mupdf.FzAaContext_bits_get, _mupdf.FzAaContext_bits_set)
    text_bits = property(_mupdf.FzAaContext_text_bits_get, _mupdf.FzAaContext_text_bits_set)
    min_line_width = property(_mupdf.FzAaContext_min_line_width_get, _mupdf.FzAaContext_min_line_width_set)
    s_num_instances = property(_mupdf.FzAaContext_s_num_instances_get, _mupdf.FzAaContext_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.FzAaContext_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzAaContext___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzAaContext___ne__(self, rhs)