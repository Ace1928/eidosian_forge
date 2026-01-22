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
class FzDefaultColorspaces(object):
    """
    Wrapper class for struct `fz_default_colorspaces`.
    Structure to hold default colorspaces.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_clone_default_colorspaces(self):
        """
        Class-aware wrapper for `::fz_clone_default_colorspaces()`.
        	Returns a reference to a newly cloned default colorspaces
        	structure.

        	The new clone may safely be altered without fear of race
        	conditions as the caller is the only reference holder.
        """
        return _mupdf.FzDefaultColorspaces_fz_clone_default_colorspaces(self)

    def fz_default_cmyk(self):
        """ Class-aware wrapper for `::fz_default_cmyk()`."""
        return _mupdf.FzDefaultColorspaces_fz_default_cmyk(self)

    def fz_default_gray(self):
        """
        Class-aware wrapper for `::fz_default_gray()`.
        	Retrieve default colorspaces (typically page local).

        	If default_cs is non NULL, the default is retrieved from there,
        	otherwise the global default is retrieved.

        	These return borrowed references that should not be dropped,
        	unless they are kept first.
        """
        return _mupdf.FzDefaultColorspaces_fz_default_gray(self)

    def fz_default_output_intent(self):
        """ Class-aware wrapper for `::fz_default_output_intent()`."""
        return _mupdf.FzDefaultColorspaces_fz_default_output_intent(self)

    def fz_default_rgb(self):
        """ Class-aware wrapper for `::fz_default_rgb()`."""
        return _mupdf.FzDefaultColorspaces_fz_default_rgb(self)

    def fz_set_default_cmyk(self, cs):
        """ Class-aware wrapper for `::fz_set_default_cmyk()`."""
        return _mupdf.FzDefaultColorspaces_fz_set_default_cmyk(self, cs)

    def fz_set_default_gray(self, cs):
        """
        Class-aware wrapper for `::fz_set_default_gray()`.
        	Set new defaults within the default colorspace structure.

        	New references are taken to the new default, and references to
        	the old defaults dropped.

        	Never throws exceptions.
        """
        return _mupdf.FzDefaultColorspaces_fz_set_default_gray(self, cs)

    def fz_set_default_output_intent(self, cs):
        """ Class-aware wrapper for `::fz_set_default_output_intent()`."""
        return _mupdf.FzDefaultColorspaces_fz_set_default_output_intent(self, cs)

    def fz_set_default_rgb(self, cs):
        """ Class-aware wrapper for `::fz_set_default_rgb()`."""
        return _mupdf.FzDefaultColorspaces_fz_set_default_rgb(self, cs)

    def pdf_update_default_colorspaces(self, res):
        """ Class-aware wrapper for `::pdf_update_default_colorspaces()`."""
        return _mupdf.FzDefaultColorspaces_pdf_update_default_colorspaces(self, res)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_default_colorspaces()`.
        		Create a new default colorspace structure with values inherited
        		from the context, and return a reference to it.

        		These can be overridden using fz_set_default_xxxx.

        		These should not be overridden while more than one caller has
        		the reference for fear of race conditions.

        		The caller should drop this reference once finished with it.


        |

        *Overload 2:*
         Copy constructor using `fz_keep_default_colorspaces()`.

        |

        *Overload 3:*
         Constructor using raw copy of pre-existing `::fz_default_colorspaces`.
        """
        _mupdf.FzDefaultColorspaces_swiginit(self, _mupdf.new_FzDefaultColorspaces(*args))
    __swig_destroy__ = _mupdf.delete_FzDefaultColorspaces

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzDefaultColorspaces_m_internal_value(self)
    m_internal = property(_mupdf.FzDefaultColorspaces_m_internal_get, _mupdf.FzDefaultColorspaces_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzDefaultColorspaces_s_num_instances_get, _mupdf.FzDefaultColorspaces_s_num_instances_set)