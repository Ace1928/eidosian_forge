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
class FzRect(object):
    """ Wrapper class for struct `fz_rect`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    Fixed_UNIT = _mupdf.FzRect_Fixed_UNIT
    Fixed_EMPTY = _mupdf.FzRect_Fixed_EMPTY
    Fixed_INFINITE = _mupdf.FzRect_Fixed_INFINITE

    def fz_adjust_rect_for_stroke(self, stroke, ctm):
        """
        Class-aware wrapper for `::fz_adjust_rect_for_stroke()`.
        	Given a rectangle (assumed to be the bounding box for a path),
        	expand it to allow for the expansion of the bbox that would be
        	seen by stroking the path with the given stroke state and
        	transform.
        """
        return _mupdf.FzRect_fz_adjust_rect_for_stroke(self, stroke, ctm)

    def fz_contains_rect(self, b):
        """
        Class-aware wrapper for `::fz_contains_rect()`.
        	Test rectangle inclusion.

        	Return true if a entirely contains b.
        """
        return _mupdf.FzRect_fz_contains_rect(self, b)

    def fz_expand_rect(self, expand):
        """
        Class-aware wrapper for `::fz_expand_rect()`.
        	Expand a bbox by a given amount in all directions.
        """
        return _mupdf.FzRect_fz_expand_rect(self, expand)

    def fz_include_point_in_rect(self, p):
        """
        Class-aware wrapper for `::fz_include_point_in_rect()`.
        	Expand a bbox to include a given point.
        	To create a rectangle that encompasses a sequence of points, the
        	rectangle must first be set to be the empty rectangle at one of
        	the points before including the others.
        """
        return _mupdf.FzRect_fz_include_point_in_rect(self, p)

    def fz_intersect_rect(self, *args):
        """
        *Overload 1:*
         Class-aware wrapper for `::fz_intersect_rect()`.
        		Compute intersection of two rectangles.

        		Given two rectangles, update the first to be the smallest
        		axis-aligned rectangle that covers the area covered by both
        		given rectangles. If either rectangle is empty then the
        		intersection is also empty. If either rectangle is infinite
        		then the intersection is simply the non-infinite rectangle.
        		Should both rectangles be infinite, then the intersection is
        		also infinite.


        |

        *Overload 2:*
         Class-aware wrapper for `::fz_intersect_rect()`.
        		Compute intersection of two rectangles.

        		Given two rectangles, update the first to be the smallest
        		axis-aligned rectangle that covers the area covered by both
        		given rectangles. If either rectangle is empty then the
        		intersection is also empty. If either rectangle is infinite
        		then the intersection is simply the non-infinite rectangle.
        		Should both rectangles be infinite, then the intersection is
        		also infinite.
        """
        return _mupdf.FzRect_fz_intersect_rect(self, *args)

    def fz_irect_from_rect(self):
        """
        Class-aware wrapper for `::fz_irect_from_rect()`.
        	Convert a rect into the minimal bounding box
        	that covers the rectangle.

        	Coordinates in a bounding box are integers, so rounding of the
        	rects coordinates takes place. The top left corner is rounded
        	upwards and left while the bottom right corner is rounded
        	downwards and to the right.
        """
        return _mupdf.FzRect_fz_irect_from_rect(self)

    def fz_is_empty_rect(self):
        """
        Class-aware wrapper for `::fz_is_empty_rect()`.
        	Check if rectangle is empty.

        	An empty rectangle is defined as one whose area is zero.
        	All invalid rectangles are empty.
        """
        return _mupdf.FzRect_fz_is_empty_rect(self)

    def fz_is_infinite_rect(self):
        """
        Class-aware wrapper for `::fz_is_infinite_rect()`.
        	Check if rectangle is infinite.
        """
        return _mupdf.FzRect_fz_is_infinite_rect(self)

    def fz_is_valid_rect(self):
        """
        Class-aware wrapper for `::fz_is_valid_rect()`.
        	Check if rectangle is valid.
        """
        return _mupdf.FzRect_fz_is_valid_rect(self)

    def fz_new_bbox_device(self):
        """
        Class-aware wrapper for `::fz_new_bbox_device()`.
        	Create a device to compute the bounding
        	box of all marks on a page.

        	The returned bounding box will be the union of all bounding
        	boxes of all objects on a page.
        """
        return _mupdf.FzRect_fz_new_bbox_device(self)

    def fz_new_display_list(self):
        """
        Class-aware wrapper for `::fz_new_display_list()`.
        	Create an empty display list.

        	A display list contains drawing commands (text, images, etc.).
        	Use fz_new_list_device for populating the list.

        	mediabox: Bounds of the page (in points) represented by the
        	display list.
        """
        return _mupdf.FzRect_fz_new_display_list(self)

    def fz_quad_from_rect(self):
        """
        Class-aware wrapper for `::fz_quad_from_rect()`.
        	Convert a rect to a quad (losslessly).
        """
        return _mupdf.FzRect_fz_quad_from_rect(self)

    def fz_round_rect(self):
        """
        Class-aware wrapper for `::fz_round_rect()`.
        	Round rectangle coordinates.

        	Coordinates in a bounding box are integers, so rounding of the
        	rects coordinates takes place. The top left corner is rounded
        	upwards and left while the bottom right corner is rounded
        	downwards and to the right.

        	This differs from fz_irect_from_rect, in that fz_irect_from_rect
        	slavishly follows the numbers (i.e any slight over/under
        	calculations can cause whole extra pixels to be added).
        	fz_round_rect allows for a small amount of rounding error when
        	calculating the bbox.
        """
        return _mupdf.FzRect_fz_round_rect(self)

    def fz_transform_page(self, resolution, rotate):
        """
        Class-aware wrapper for `::fz_transform_page()`.
        	Create transform matrix to draw page
        	at a given resolution and rotation. Adjusts the scaling
        	factors so that the page covers whole number of
        	pixels and adjust the page origin to be at 0,0.
        """
        return _mupdf.FzRect_fz_transform_page(self, resolution, rotate)

    def fz_transform_rect(self, m):
        """
        Class-aware wrapper for `::fz_transform_rect()`.
        	Apply a transform to a rectangle.

        	After the four corner points of the axis-aligned rectangle
        	have been transformed it may not longer be axis-aligned. So a
        	new axis-aligned rectangle is created covering at least the
        	area of the transformed rectangle.

        	transform: Transformation matrix to apply. See fz_concat,
        	fz_scale and fz_rotate for how to create a matrix.

        	rect: Rectangle to be transformed. The two special cases
        	fz_empty_rect and fz_infinite_rect, may be used but are
        	returned unchanged as expected.
        """
        return _mupdf.FzRect_fz_transform_rect(self, m)

    def fz_translate_rect(self, xoff, yoff):
        """
        Class-aware wrapper for `::fz_translate_rect()`.
        	Translate bounding box.

        	Translate a bbox by a given x and y offset. Allows for overflow.
        """
        return _mupdf.FzRect_fz_translate_rect(self, xoff, yoff)

    def fz_union_rect(self, *args):
        """
        *Overload 1:*
         Class-aware wrapper for `::fz_union_rect()`.
        		Compute union of two rectangles.

        		Given two rectangles, update the first to be the smallest
        		axis-aligned rectangle that encompasses both given rectangles.
        		If either rectangle is infinite then the union is also infinite.
        		If either rectangle is empty then the union is simply the
        		non-empty rectangle. Should both rectangles be empty, then the
        		union is also empty.


        |

        *Overload 2:*
         Class-aware wrapper for `::fz_union_rect()`.
        		Compute union of two rectangles.

        		Given two rectangles, update the first to be the smallest
        		axis-aligned rectangle that encompasses both given rectangles.
        		If either rectangle is infinite then the union is also infinite.
        		If either rectangle is empty then the union is simply the
        		non-empty rectangle. Should both rectangles be empty, then the
        		union is also empty.
        """
        return _mupdf.FzRect_fz_union_rect(self, *args)

    def pdf_signature_appearance_signed(self, lang, img, left_text, right_text, include_logo):
        """ Class-aware wrapper for `::pdf_signature_appearance_signed()`."""
        return _mupdf.FzRect_pdf_signature_appearance_signed(self, lang, img, left_text, right_text, include_logo)

    def pdf_signature_appearance_unsigned(self, lang):
        """ Class-aware wrapper for `::pdf_signature_appearance_unsigned()`."""
        return _mupdf.FzRect_pdf_signature_appearance_unsigned(self, lang)

    def transform(self, m):
        """ Transforms *this using fz_transform_rect() with <m>."""
        return _mupdf.FzRect_transform(self, m)

    def contains(self, *args):
        """
        *Overload 1:*
        Convenience method using fz_contains_rect().

        |

        *Overload 2:*
        Uses fz_contains_rect(*this, rhs).
        """
        return _mupdf.FzRect_contains(self, *args)

    def is_empty(self):
        """ Uses fz_is_empty_rect()."""
        return _mupdf.FzRect_is_empty(self)

    def union_(self, rhs):
        """ Updates *this using fz_union_rect()."""
        return _mupdf.FzRect_union_(self, rhs)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_bound_display_list()`.
        		Return the bounding box of the page recorded in a display list.


        |

        *Overload 2:*
         Constructor using `fz_rect_from_irect()`.
        		Convert a bbox into a rect.

        		For our purposes, a rect can represent all the values we meet in
        		a bbox, so nothing can go wrong.

        		rect: A place to store the generated rectangle.

        		bbox: The bbox to convert.

        		Returns rect (updated).


        |

        *Overload 3:*
         Constructor using `fz_rect_from_quad()`.
        		Convert a quad to the smallest rect that covers it.


        |

        *Overload 4:*
         Constructor using `fz_transform_rect()`.
        		Apply a transform to a rectangle.

        		After the four corner points of the axis-aligned rectangle
        		have been transformed it may not longer be axis-aligned. So a
        		new axis-aligned rectangle is created covering at least the
        		area of the transformed rectangle.

        		transform: Transformation matrix to apply. See fz_concat,
        		fz_scale and fz_rotate for how to create a matrix.

        		rect: Rectangle to be transformed. The two special cases
        		fz_empty_rect and fz_infinite_rect, may be used but are
        		returned unchanged as expected.


        |

        *Overload 5:*
         Construct from specified values.

        |

        *Overload 6:*
         Copy constructor using plain copy.

        |

        *Overload 7:*
         Construct from fz_unit_rect, fz_empty_rect or fz_infinite_rect.

        |

        *Overload 8:*
         We use default copy constructor and operator=.  Default constructor, sets each member to default value.

        |

        *Overload 9:*
         Constructor using raw copy of pre-existing `::fz_rect`.

        |

        *Overload 10:*
         Constructor using raw copy of pre-existing `::fz_rect`.
        """
        _mupdf.FzRect_swiginit(self, _mupdf.new_FzRect(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.FzRect_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_FzRect
    x0 = property(_mupdf.FzRect_x0_get, _mupdf.FzRect_x0_set)
    y0 = property(_mupdf.FzRect_y0_get, _mupdf.FzRect_y0_set)
    x1 = property(_mupdf.FzRect_x1_get, _mupdf.FzRect_x1_set)
    y1 = property(_mupdf.FzRect_y1_get, _mupdf.FzRect_y1_set)
    s_num_instances = property(_mupdf.FzRect_s_num_instances_get, _mupdf.FzRect_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.FzRect_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzRect___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzRect___ne__(self, rhs)