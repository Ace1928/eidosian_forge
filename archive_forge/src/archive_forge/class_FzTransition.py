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
class FzTransition(object):
    """ Wrapper class for struct `fz_transition`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        We use default copy constructor and operator=.  Default constructor, sets each member to default value.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_transition`.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::fz_transition`.
        """
        _mupdf.FzTransition_swiginit(self, _mupdf.new_FzTransition(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.FzTransition_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_FzTransition
    type = property(_mupdf.FzTransition_type_get, _mupdf.FzTransition_type_set)
    duration = property(_mupdf.FzTransition_duration_get, _mupdf.FzTransition_duration_set)
    vertical = property(_mupdf.FzTransition_vertical_get, _mupdf.FzTransition_vertical_set)
    outwards = property(_mupdf.FzTransition_outwards_get, _mupdf.FzTransition_outwards_set)
    direction = property(_mupdf.FzTransition_direction_get, _mupdf.FzTransition_direction_set)
    state0 = property(_mupdf.FzTransition_state0_get, _mupdf.FzTransition_state0_set)
    state1 = property(_mupdf.FzTransition_state1_get, _mupdf.FzTransition_state1_set)
    s_num_instances = property(_mupdf.FzTransition_s_num_instances_get, _mupdf.FzTransition_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.FzTransition_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzTransition___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzTransition___ne__(self, rhs)