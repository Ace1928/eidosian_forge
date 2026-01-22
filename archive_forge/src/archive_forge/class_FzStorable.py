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
class FzStorable(object):
    """
    Wrapper class for struct `fz_storable`.
    Resource store

    MuPDF stores decoded "objects" into a store for potential reuse.
    If the size of the store gets too big, objects stored within it
    can be evicted and freed to recover space. When MuPDF comes to
    decode such an object, it will check to see if a version of this
    object is already in the store - if it is, it will simply reuse
    it. If not, it will decode it and place it into the store.

    All objects that can be placed into the store are derived from
    the fz_storable type (i.e. this should be the first component of
    the objects structure). This allows for consistent (thread safe)
    reference counting, and includes a function that will be called
    to free the object as soon as the reference count reaches zero.

    Most objects offer fz_keep_XXXX/fz_drop_XXXX functions derived
    from fz_keep_storable/fz_drop_storable. Creation of such objects
    includes a call to FZ_INIT_STORABLE to set up the fz_storable
    header.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        Copy constructor using `fz_keep_storable()`.

        |

        *Overload 2:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::fz_storable`.
        """
        _mupdf.FzStorable_swiginit(self, _mupdf.new_FzStorable(*args))
    __swig_destroy__ = _mupdf.delete_FzStorable

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzStorable_m_internal_value(self)
    m_internal = property(_mupdf.FzStorable_m_internal_get, _mupdf.FzStorable_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzStorable_s_num_instances_get, _mupdf.FzStorable_s_num_instances_set)