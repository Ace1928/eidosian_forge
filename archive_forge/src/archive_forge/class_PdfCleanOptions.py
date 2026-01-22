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
class PdfCleanOptions(object):
    """ Wrapper class for struct `pdf_clean_options`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def write_opwd_utf8_set(self, text):
        """ Copies <text> into write.opwd_utf8[]."""
        return _mupdf.PdfCleanOptions_write_opwd_utf8_set(self, text)

    def write_upwd_utf8_set(self, text):
        """ Copies <text> into upwd_utf8[]."""
        return _mupdf.PdfCleanOptions_write_upwd_utf8_set(self, text)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, makes copy of pdf_default_write_options.

        |

        *Overload 2:*
        Copy constructor using raw memcopy().

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::pdf_clean_options`.

        |

        *Overload 4:*
        Constructor using raw copy of pre-existing `::pdf_clean_options`.
        """
        _mupdf.PdfCleanOptions_swiginit(self, _mupdf.new_PdfCleanOptions(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.PdfCleanOptions_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_PdfCleanOptions
    write = property(_mupdf.PdfCleanOptions_write_get, _mupdf.PdfCleanOptions_write_set)
    image = property(_mupdf.PdfCleanOptions_image_get, _mupdf.PdfCleanOptions_image_set)
    subset_fonts = property(_mupdf.PdfCleanOptions_subset_fonts_get, _mupdf.PdfCleanOptions_subset_fonts_set)
    s_num_instances = property(_mupdf.PdfCleanOptions_s_num_instances_get, _mupdf.PdfCleanOptions_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.PdfCleanOptions_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.PdfCleanOptions___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.PdfCleanOptions___ne__(self, rhs)