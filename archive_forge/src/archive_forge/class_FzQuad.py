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
class FzQuad(object):
    """
    Wrapper class for struct `fz_quad`.
    A representation for a region defined by 4 points.

    The significant difference between quads and rects is that
    the edges of quads are not axis aligned.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_is_quad_inside_quad(self, haystack):
        """
        Class-aware wrapper for `::fz_is_quad_inside_quad()`.
        	Inclusion test for quad in quad.

        	This may break down if quads are not 'well formed'.
        """
        return _mupdf.FzQuad_fz_is_quad_inside_quad(self, haystack)

    def fz_is_quad_intersecting_quad(self, b):
        """
        Class-aware wrapper for `::fz_is_quad_intersecting_quad()`.
        	Intersection test for quads.

        	This may break down if quads are not 'well formed'.
        """
        return _mupdf.FzQuad_fz_is_quad_intersecting_quad(self, b)

    def fz_rect_from_quad(self):
        """
        Class-aware wrapper for `::fz_rect_from_quad()`.
        	Convert a quad to the smallest rect that covers it.
        """
        return _mupdf.FzQuad_fz_rect_from_quad(self)

    def fz_transform_quad(self, m):
        """
        Class-aware wrapper for `::fz_transform_quad()`.
        	Transform a quad by a matrix.
        """
        return _mupdf.FzQuad_fz_transform_quad(self, m)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_quad_from_rect()`.
        		Convert a rect to a quad (losslessly).


        |

        *Overload 2:*
         Constructor using `fz_transform_quad()`.
        		Transform a quad by a matrix.


        |

        *Overload 3:*
         We use default copy constructor and operator=.  Default constructor, sets each member to default value.

        |

        *Overload 4:*
         Constructor using raw copy of pre-existing `::fz_quad`.

        |

        *Overload 5:*
         Constructor using raw copy of pre-existing `::fz_quad`.
        """
        _mupdf.FzQuad_swiginit(self, _mupdf.new_FzQuad(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.FzQuad_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_FzQuad
    ul = property(_mupdf.FzQuad_ul_get, _mupdf.FzQuad_ul_set)
    ur = property(_mupdf.FzQuad_ur_get, _mupdf.FzQuad_ur_set)
    ll = property(_mupdf.FzQuad_ll_get, _mupdf.FzQuad_ll_set)
    lr = property(_mupdf.FzQuad_lr_get, _mupdf.FzQuad_lr_set)
    s_num_instances = property(_mupdf.FzQuad_s_num_instances_get, _mupdf.FzQuad_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.FzQuad_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzQuad___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzQuad___ne__(self, rhs)