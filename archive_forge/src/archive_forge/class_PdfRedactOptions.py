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
class PdfRedactOptions(object):
    """ Wrapper class for struct `pdf_redact_options`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        We use default copy constructor and operator=.  Default constructor, sets each member to default value.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::pdf_redact_options`.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::pdf_redact_options`.
        """
        _mupdf.PdfRedactOptions_swiginit(self, _mupdf.new_PdfRedactOptions(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.PdfRedactOptions_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_PdfRedactOptions
    black_boxes = property(_mupdf.PdfRedactOptions_black_boxes_get, _mupdf.PdfRedactOptions_black_boxes_set)
    image_method = property(_mupdf.PdfRedactOptions_image_method_get, _mupdf.PdfRedactOptions_image_method_set)
    line_art = property(_mupdf.PdfRedactOptions_line_art_get, _mupdf.PdfRedactOptions_line_art_set)
    s_num_instances = property(_mupdf.PdfRedactOptions_s_num_instances_get, _mupdf.PdfRedactOptions_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.PdfRedactOptions_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.PdfRedactOptions___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.PdfRedactOptions___ne__(self, rhs)