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
class FzTextDecoder(object):
    """
     Wrapper class for struct `fz_text_decoder`. Not copyable or assignable.
    A text decoder (to read arbitrary encodings and convert to unicode).
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_init_text_decoder(self, encoding):
        """ Class-aware wrapper for `::fz_init_text_decoder()`."""
        return _mupdf.FzTextDecoder_fz_init_text_decoder(self, encoding)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_text_decoder`.
        """
        _mupdf.FzTextDecoder_swiginit(self, _mupdf.new_FzTextDecoder(*args))
    __swig_destroy__ = _mupdf.delete_FzTextDecoder

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzTextDecoder_m_internal_value(self)
    m_internal = property(_mupdf.FzTextDecoder_m_internal_get, _mupdf.FzTextDecoder_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzTextDecoder_s_num_instances_get, _mupdf.FzTextDecoder_s_num_instances_set)