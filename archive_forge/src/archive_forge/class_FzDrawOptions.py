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
class FzDrawOptions(object):
    """
    Wrapper class for struct `fz_draw_options`. Not copyable or assignable.
    struct fz_draw_options: Options for creating a pixmap and draw
    device.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_parse_draw_options()`.
        		Parse draw device options from a comma separated key-value string.


        |

        *Overload 2:*
         Default constructor, sets each member to default value.

        |

        *Overload 3:*
         Constructor using raw copy of pre-existing `::fz_draw_options`.

        |

        *Overload 4:*
         Constructor using raw copy of pre-existing `::fz_draw_options`.
        """
        _mupdf.FzDrawOptions_swiginit(self, _mupdf.new_FzDrawOptions(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.FzDrawOptions_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_FzDrawOptions
    rotate = property(_mupdf.FzDrawOptions_rotate_get, _mupdf.FzDrawOptions_rotate_set)
    x_resolution = property(_mupdf.FzDrawOptions_x_resolution_get, _mupdf.FzDrawOptions_x_resolution_set)
    y_resolution = property(_mupdf.FzDrawOptions_y_resolution_get, _mupdf.FzDrawOptions_y_resolution_set)
    width = property(_mupdf.FzDrawOptions_width_get, _mupdf.FzDrawOptions_width_set)
    height = property(_mupdf.FzDrawOptions_height_get, _mupdf.FzDrawOptions_height_set)
    colorspace = property(_mupdf.FzDrawOptions_colorspace_get, _mupdf.FzDrawOptions_colorspace_set)
    alpha = property(_mupdf.FzDrawOptions_alpha_get, _mupdf.FzDrawOptions_alpha_set)
    graphics = property(_mupdf.FzDrawOptions_graphics_get, _mupdf.FzDrawOptions_graphics_set)
    text = property(_mupdf.FzDrawOptions_text_get, _mupdf.FzDrawOptions_text_set)
    s_num_instances = property(_mupdf.FzDrawOptions_s_num_instances_get, _mupdf.FzDrawOptions_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.FzDrawOptions_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzDrawOptions___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzDrawOptions___ne__(self, rhs)