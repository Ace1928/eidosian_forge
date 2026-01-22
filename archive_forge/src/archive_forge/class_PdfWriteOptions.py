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
class PdfWriteOptions(object):
    """ Wrapper class for struct `pdf_write_options`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def pdf_parse_write_options(self, args):
        """ We use default copy constructor and operator=.  Class-aware wrapper for `::pdf_parse_write_options()`."""
        return _mupdf.PdfWriteOptions_pdf_parse_write_options(self, args)

    def opwd_utf8_set_value(self, text):
        """ Copies <text> into opwd_utf8[]."""
        return _mupdf.PdfWriteOptions_opwd_utf8_set_value(self, text)

    def upwd_utf8_set_value(self, text):
        """ Copies <text> into upwd_utf8[]."""
        return _mupdf.PdfWriteOptions_upwd_utf8_set_value(self, text)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, makes copy of pdf_default_write_options.

        |

        *Overload 2:*
        Copy constructor using raw memcopy().

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::pdf_write_options`.

        |

        *Overload 4:*
        Constructor using raw copy of pre-existing `::pdf_write_options`.
        """
        _mupdf.PdfWriteOptions_swiginit(self, _mupdf.new_PdfWriteOptions(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.PdfWriteOptions_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_PdfWriteOptions
    do_incremental = property(_mupdf.PdfWriteOptions_do_incremental_get, _mupdf.PdfWriteOptions_do_incremental_set)
    do_pretty = property(_mupdf.PdfWriteOptions_do_pretty_get, _mupdf.PdfWriteOptions_do_pretty_set)
    do_ascii = property(_mupdf.PdfWriteOptions_do_ascii_get, _mupdf.PdfWriteOptions_do_ascii_set)
    do_compress = property(_mupdf.PdfWriteOptions_do_compress_get, _mupdf.PdfWriteOptions_do_compress_set)
    do_compress_images = property(_mupdf.PdfWriteOptions_do_compress_images_get, _mupdf.PdfWriteOptions_do_compress_images_set)
    do_compress_fonts = property(_mupdf.PdfWriteOptions_do_compress_fonts_get, _mupdf.PdfWriteOptions_do_compress_fonts_set)
    do_decompress = property(_mupdf.PdfWriteOptions_do_decompress_get, _mupdf.PdfWriteOptions_do_decompress_set)
    do_garbage = property(_mupdf.PdfWriteOptions_do_garbage_get, _mupdf.PdfWriteOptions_do_garbage_set)
    do_linear = property(_mupdf.PdfWriteOptions_do_linear_get, _mupdf.PdfWriteOptions_do_linear_set)
    do_clean = property(_mupdf.PdfWriteOptions_do_clean_get, _mupdf.PdfWriteOptions_do_clean_set)
    do_sanitize = property(_mupdf.PdfWriteOptions_do_sanitize_get, _mupdf.PdfWriteOptions_do_sanitize_set)
    do_appearance = property(_mupdf.PdfWriteOptions_do_appearance_get, _mupdf.PdfWriteOptions_do_appearance_set)
    do_encrypt = property(_mupdf.PdfWriteOptions_do_encrypt_get, _mupdf.PdfWriteOptions_do_encrypt_set)
    dont_regenerate_id = property(_mupdf.PdfWriteOptions_dont_regenerate_id_get, _mupdf.PdfWriteOptions_dont_regenerate_id_set)
    permissions = property(_mupdf.PdfWriteOptions_permissions_get, _mupdf.PdfWriteOptions_permissions_set)
    opwd_utf8 = property(_mupdf.PdfWriteOptions_opwd_utf8_get, _mupdf.PdfWriteOptions_opwd_utf8_set)
    upwd_utf8 = property(_mupdf.PdfWriteOptions_upwd_utf8_get, _mupdf.PdfWriteOptions_upwd_utf8_set)
    do_snapshot = property(_mupdf.PdfWriteOptions_do_snapshot_get, _mupdf.PdfWriteOptions_do_snapshot_set)
    do_preserve_metadata = property(_mupdf.PdfWriteOptions_do_preserve_metadata_get, _mupdf.PdfWriteOptions_do_preserve_metadata_set)
    do_use_objstms = property(_mupdf.PdfWriteOptions_do_use_objstms_get, _mupdf.PdfWriteOptions_do_use_objstms_set)
    compression_effort = property(_mupdf.PdfWriteOptions_compression_effort_get, _mupdf.PdfWriteOptions_compression_effort_set)
    s_num_instances = property(_mupdf.PdfWriteOptions_s_num_instances_get, _mupdf.PdfWriteOptions_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.PdfWriteOptions_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.PdfWriteOptions___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.PdfWriteOptions___ne__(self, rhs)