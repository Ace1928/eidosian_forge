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
class FzDisplayList(object):
    """
    Wrapper class for struct `fz_display_list`.
    fz_display_list is a list containing drawing commands (text,
    images, etc.). The intent is two-fold: as a caching-mechanism
    to reduce parsing of a page, and to be used as a data
    structure in multi-threading where one thread parses the page
    and another renders pages.

    Create a display list with fz_new_display_list, hand it over to
    fz_new_list_device to have it populated, and later replay the
    list (once or many times) by calling fz_run_display_list. When
    the list is no longer needed drop it with fz_drop_display_list.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    @staticmethod
    def fz_new_display_list_from_page_contents(page):
        """
        Class-aware wrapper for `::fz_new_display_list_from_page_contents()`.
        	Create a display list from page contents (no annotations).

        	Ownership of the display list is returned to the caller.
        """
        return _mupdf.FzDisplayList_fz_new_display_list_from_page_contents(page)

    def fz_bound_display_list(self):
        """
        Class-aware wrapper for `::fz_bound_display_list()`.
        	Return the bounding box of the page recorded in a display list.
        """
        return _mupdf.FzDisplayList_fz_bound_display_list(self)

    def fz_display_list_is_empty(self):
        """
        Class-aware wrapper for `::fz_display_list_is_empty()`.
        	Check for a display list being empty

        	list: The list to check.

        	Returns true if empty, false otherwise.
        """
        return _mupdf.FzDisplayList_fz_display_list_is_empty(self)

    def fz_fill_pixmap_from_display_list(self, ctm, pix):
        """ Class-aware wrapper for `::fz_fill_pixmap_from_display_list()`."""
        return _mupdf.FzDisplayList_fz_fill_pixmap_from_display_list(self, ctm, pix)

    def fz_new_buffer_from_display_list(self, options):
        """ Class-aware wrapper for `::fz_new_buffer_from_display_list()`."""
        return _mupdf.FzDisplayList_fz_new_buffer_from_display_list(self, options)

    def fz_new_list_device(self):
        """
        Class-aware wrapper for `::fz_new_list_device()`.
        	Create a rendering device for a display list.

        	When the device is rendering a page it will populate the
        	display list with drawing commands (text, images, etc.). The
        	display list can later be reused to render a page many times
        	without having to re-interpret the page from the document file
        	for each rendering. Once the device is no longer needed, free
        	it with fz_drop_device.

        	list: A display list that the list device takes a reference to.
        """
        return _mupdf.FzDisplayList_fz_new_list_device(self)

    def fz_new_pixmap_from_display_list(self, ctm, cs, alpha):
        """
        Class-aware wrapper for `::fz_new_pixmap_from_display_list()`.
        	Render the page to a pixmap using the transform and colorspace.

        	Ownership of the pixmap is returned to the caller.
        """
        return _mupdf.FzDisplayList_fz_new_pixmap_from_display_list(self, ctm, cs, alpha)

    def fz_new_pixmap_from_display_list_with_separations(self, ctm, cs, seps, alpha):
        """
        Class-aware wrapper for `::fz_new_pixmap_from_display_list_with_separations()`.
        	Render the page contents with control over spot colors.

        	Ownership of the pixmap is returned to the caller.
        """
        return _mupdf.FzDisplayList_fz_new_pixmap_from_display_list_with_separations(self, ctm, cs, seps, alpha)

    def fz_run_display_list(self, dev, ctm, scissor, cookie):
        """
        Class-aware wrapper for `::fz_run_display_list()`.
        	(Re)-run a display list through a device.

        	list: A display list, created by fz_new_display_list and
        	populated with objects from a page by running fz_run_page on a
        	device obtained from fz_new_list_device.

        	ctm: Transform to apply to display list contents. May include
        	for example scaling and rotation, see fz_scale, fz_rotate and
        	fz_concat. Set to fz_identity if no transformation is desired.

        	scissor: Only the part of the contents of the display list
        	visible within this area will be considered when the list is
        	run through the device. This does not imply for tile objects
        	contained in the display list.

        	cookie: Communication mechanism between caller and library
        	running the page. Intended for multi-threaded applications,
        	while single-threaded applications set cookie to NULL. The
        	caller may abort an ongoing page run. Cookie also communicates
        	progress information back to the caller. The fields inside
        	cookie are continually updated while the page is being run.
        """
        return _mupdf.FzDisplayList_fz_run_display_list(self, dev, ctm, scissor, cookie)

    def fz_search_display_list(self, needle, hit_mark, hit_bbox, hit_max):
        """
        Class-aware wrapper for `::fz_search_display_list()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_search_display_list(const char *needle, ::fz_quad *hit_bbox, int hit_max)` => `(int, int hit_mark)`
        """
        return _mupdf.FzDisplayList_fz_search_display_list(self, needle, hit_mark, hit_bbox, hit_max)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_display_list()`.
        		Create an empty display list.

        		A display list contains drawing commands (text, images, etc.).
        		Use fz_new_list_device for populating the list.

        		mediabox: Bounds of the page (in points) represented by the
        		display list.


        |

        *Overload 2:*
         Constructor using `fz_new_display_list_from_page()`.
        		Create a display list.

        		Ownership of the display list is returned to the caller.


        |

        *Overload 3:*
         Constructor using `fz_new_display_list_from_page_number()`.

        |

        *Overload 4:*
         Constructor using `fz_new_display_list_from_svg()`.
        		Parse an SVG document into a display-list.


        |

        *Overload 5:*
         Constructor using `fz_new_display_list_from_svg_xml()`.
        		Parse an SVG document into a display-list.


        |

        *Overload 6:*
         Constructor using `pdf_new_display_list_from_annot()`.

        |

        *Overload 7:*
         Copy constructor using `fz_keep_display_list()`.

        |

        *Overload 8:*
         Default constructor, sets `m_internal` to null.

        |

        *Overload 9:*
         Constructor using raw copy of pre-existing `::fz_display_list`.
        """
        _mupdf.FzDisplayList_swiginit(self, _mupdf.new_FzDisplayList(*args))
    __swig_destroy__ = _mupdf.delete_FzDisplayList

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzDisplayList_m_internal_value(self)
    m_internal = property(_mupdf.FzDisplayList_m_internal_get, _mupdf.FzDisplayList_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzDisplayList_s_num_instances_get, _mupdf.FzDisplayList_s_num_instances_set)