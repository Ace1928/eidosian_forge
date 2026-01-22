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
class FzPclmOptions(object):
    """
    Wrapper class for struct `fz_pclm_options`. Not copyable or assignable.
    PCLm output
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_parse_pclm_options()`.
        		Parse PCLm options.

        		Currently defined options and values are as follows:

        			compression=none: No compression
        			compression=flate: Flate compression
        			strip-height=n: Strip height (default 16)


        |

        *Overload 2:*
         Construct using fz_parse_pclm_options().

        |

        *Overload 3:*
         Default constructor, sets `m_internal` to null.

        |

        *Overload 4:*
         Constructor using raw copy of pre-existing `::fz_pclm_options`.
        """
        _mupdf.FzPclmOptions_swiginit(self, _mupdf.new_FzPclmOptions(*args))
    __swig_destroy__ = _mupdf.delete_FzPclmOptions

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzPclmOptions_m_internal_value(self)
    m_internal = property(_mupdf.FzPclmOptions_m_internal_get, _mupdf.FzPclmOptions_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzPclmOptions_s_num_instances_get, _mupdf.FzPclmOptions_s_num_instances_set)