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
class FzStrokeState(object):
    """ Wrapper class for struct `fz_stroke_state`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_clone_stroke_state(self):
        """
        Class-aware wrapper for `::fz_clone_stroke_state()`.
        	Create an identical stroke_state structure and return a
        	reference to it.

        	stroke: The stroke state reference to clone.

        	Exceptions may be thrown in the event of a failure to
        	allocate.
        """
        return _mupdf.FzStrokeState_fz_clone_stroke_state(self)

    def fz_unshare_stroke_state(self):
        """
        Class-aware wrapper for `::fz_unshare_stroke_state()`.
        	Given a reference to a (possibly) shared stroke_state structure,
        	return a reference to an equivalent stroke_state structure
        	that is guaranteed to be unshared (i.e. one that can
        	safely be modified).

        	shared: The reference to a (possibly) shared structure
        	to unshare. Ownership of this reference is passed in
        	to this function, even in the case of exceptions being
        	thrown.

        	Exceptions may be thrown in the event of failure to
        	allocate if required.
        """
        return _mupdf.FzStrokeState_fz_unshare_stroke_state(self)

    def fz_unshare_stroke_state_with_dash_len(self, len):
        """
        Class-aware wrapper for `::fz_unshare_stroke_state_with_dash_len()`.
        	Given a reference to a (possibly) shared stroke_state structure,
        	return a reference to a stroke_state structure (with room for a
        	given amount of dash data) that is guaranteed to be unshared
        	(i.e. one that can safely be modified).

        	shared: The reference to a (possibly) shared structure
        	to unshare. Ownership of this reference is passed in
        	to this function, even in the case of exceptions being
        	thrown.

        	Exceptions may be thrown in the event of failure to
        	allocate if required.
        """
        return _mupdf.FzStrokeState_fz_unshare_stroke_state_with_dash_len(self, len)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_stroke_state()`.
        		Create a new (empty) stroke state structure (with no dash
        		data) and return a reference to it.

        		Throws exception on failure to allocate.


        |

        *Overload 2:*
         Constructor using `fz_new_stroke_state_with_dash_len()`.
        		Create a new (empty) stroke state structure, with room for
        		dash data of the given length, and return a reference to it.

        		len: The number of dash elements to allow room for.

        		Throws exception on failure to allocate.


        |

        *Overload 3:*
         Copy constructor using `fz_keep_stroke_state()`.

        |

        *Overload 4:*
         Constructor using raw copy of pre-existing `::fz_stroke_state`.
        """
        _mupdf.FzStrokeState_swiginit(self, _mupdf.new_FzStrokeState(*args))
    __swig_destroy__ = _mupdf.delete_FzStrokeState

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzStrokeState_m_internal_value(self)
    m_internal = property(_mupdf.FzStrokeState_m_internal_get, _mupdf.FzStrokeState_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzStrokeState_s_num_instances_get, _mupdf.FzStrokeState_s_num_instances_set)