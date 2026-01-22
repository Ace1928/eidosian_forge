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
class FzStextOptions(object):
    """
    Wrapper class for struct `fz_stext_options`.
    Options for creating structured text.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_parse_stext_options(self, string):
        """
        Class-aware wrapper for `::fz_parse_stext_options()`.
        	Parse stext device options from a comma separated key-value
        	string.
        """
        return _mupdf.FzStextOptions_fz_parse_stext_options(self, string)

    def __init__(self, *args):
        """
        *Overload 1:*
        Construct with .flags set to <flags>.

        |

        *Overload 2:*
        We use default copy constructor and operator=.  Default constructor, sets each member to default value.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::fz_stext_options`.

        |

        *Overload 4:*
        Constructor using raw copy of pre-existing `::fz_stext_options`.
        """
        _mupdf.FzStextOptions_swiginit(self, _mupdf.new_FzStextOptions(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.FzStextOptions_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_FzStextOptions
    flags = property(_mupdf.FzStextOptions_flags_get, _mupdf.FzStextOptions_flags_set)
    scale = property(_mupdf.FzStextOptions_scale_get, _mupdf.FzStextOptions_scale_set)
    s_num_instances = property(_mupdf.FzStextOptions_s_num_instances_get, _mupdf.FzStextOptions_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.FzStextOptions_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzStextOptions___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzStextOptions___ne__(self, rhs)