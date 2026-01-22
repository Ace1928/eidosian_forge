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
class PdfSanitizeFilterOptions(object):
    """ Wrapper class for struct `pdf_sanitize_filter_options`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        We use default copy constructor and operator=.  Default constructor, sets each member to default value.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::pdf_sanitize_filter_options`.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::pdf_sanitize_filter_options`.
        """
        _mupdf.PdfSanitizeFilterOptions_swiginit(self, _mupdf.new_PdfSanitizeFilterOptions(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.PdfSanitizeFilterOptions_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_PdfSanitizeFilterOptions
    opaque = property(_mupdf.PdfSanitizeFilterOptions_opaque_get, _mupdf.PdfSanitizeFilterOptions_opaque_set)
    image_filter = property(_mupdf.PdfSanitizeFilterOptions_image_filter_get, _mupdf.PdfSanitizeFilterOptions_image_filter_set)
    text_filter = property(_mupdf.PdfSanitizeFilterOptions_text_filter_get, _mupdf.PdfSanitizeFilterOptions_text_filter_set)
    after_text_object = property(_mupdf.PdfSanitizeFilterOptions_after_text_object_get, _mupdf.PdfSanitizeFilterOptions_after_text_object_set)
    culler = property(_mupdf.PdfSanitizeFilterOptions_culler_get, _mupdf.PdfSanitizeFilterOptions_culler_set)
    s_num_instances = property(_mupdf.PdfSanitizeFilterOptions_s_num_instances_get, _mupdf.PdfSanitizeFilterOptions_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.PdfSanitizeFilterOptions_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.PdfSanitizeFilterOptions___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.PdfSanitizeFilterOptions___ne__(self, rhs)