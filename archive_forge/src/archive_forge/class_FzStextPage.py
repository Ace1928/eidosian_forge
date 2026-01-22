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
class FzStextPage(object):
    """
    Wrapper class for struct `fz_stext_page`. Not copyable or assignable.
    A text page is a list of blocks, together with an overall
    bounding box.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_highlight_selection(self, a, b, quads, max_quads):
        """
        Class-aware wrapper for `::fz_highlight_selection()`.
        	Return a list of quads to highlight lines inside the selection
        	points.
        """
        return _mupdf.FzStextPage_fz_highlight_selection(self, a, b, quads, max_quads)

    def fz_highlight_selection2(self, a, b, max_quads):
        """
        Class-aware wrapper for `::fz_highlight_selection2()`.
        C++ alternative to fz_highlight_selection() that returns quads in a
        std::vector.
        """
        return _mupdf.FzStextPage_fz_highlight_selection2(self, a, b, max_quads)

    def fz_new_buffer_from_stext_page(self):
        """
        Class-aware wrapper for `::fz_new_buffer_from_stext_page()`.
        	Convert structured text into plain text.
        """
        return _mupdf.FzStextPage_fz_new_buffer_from_stext_page(self)

    def fz_new_stext_device(self, options):
        """
        Class-aware wrapper for `::fz_new_stext_device()`.
        	Create a device to extract the text on a page.

        	Gather the text on a page into blocks and lines.

        	The reading order is taken from the order the text is drawn in
        	the source file, so may not be accurate.

        	page: The text page to which content should be added. This will
        	usually be a newly created (empty) text page, but it can be one
        	containing data already (for example when merging multiple
        	pages, or watermarking).

        	options: Options to configure the stext device.
        """
        return _mupdf.FzStextPage_fz_new_stext_device(self, options)

    def fz_search_stext_page(self, needle, hit_mark, hit_bbox, hit_max):
        """
        Class-aware wrapper for `::fz_search_stext_page()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_search_stext_page(const char *needle, ::fz_quad *hit_bbox, int hit_max)` => `(int, int hit_mark)`

        	Search for occurrence of 'needle' in text page.

        	Return the number of hits and store hit quads in the passed in
        	array.

        	NOTE: This is an experimental interface and subject to change
        	without notice.
        """
        return _mupdf.FzStextPage_fz_search_stext_page(self, needle, hit_mark, hit_bbox, hit_max)

    def fz_snap_selection(self, ap, bp, mode):
        """ Class-aware wrapper for `::fz_snap_selection()`."""
        return _mupdf.FzStextPage_fz_snap_selection(self, ap, bp, mode)

    def fz_copy_selection(self, a, b, crlf):
        """ Wrapper for fz_copy_selection() that returns std::string."""
        return _mupdf.FzStextPage_fz_copy_selection(self, a, b, crlf)

    def fz_copy_rectangle(self, area, crlf):
        """ Wrapper for fz_copy_rectangle() that returns a std::string."""
        return _mupdf.FzStextPage_fz_copy_rectangle(self, area, crlf)

    def search_stext_page(self, needle, hit_mark, max_quads):
        """ Wrapper for fz_search_stext_page() that returns std::vector of Quads."""
        return _mupdf.FzStextPage_search_stext_page(self, needle, hit_mark, max_quads)

    def begin(self):
        """ Used for iteration over linked list of FzStextBlock items starting at fz_stext_block::first_block."""
        return _mupdf.FzStextPage_begin(self)

    def end(self):
        """ Used for iteration over linked list of FzStextBlock items starting at fz_stext_block::first_block."""
        return _mupdf.FzStextPage_end(self)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_stext_page()`.
        		Create an empty text page.

        		The text page is filled out by the text device to contain the
        		blocks and lines of text on the page.

        		mediabox: optional mediabox information.


        |

        *Overload 2:*
         Constructor using `fz_new_stext_page_from_chapter_page_number()`.

        |

        *Overload 3:*
         Constructor using `fz_new_stext_page_from_display_list()`.

        |

        *Overload 4:*
         Constructor using `fz_new_stext_page_from_page()`.
        		Extract text from page.

        		Ownership of the fz_stext_page is returned to the caller.


        |

        *Overload 5:*
         Constructor using `fz_new_stext_page_from_page_number()`.

        |

        *Overload 6:*
         Constructor using `pdf_new_stext_page_from_annot()`.

        |

        *Overload 7:*
         Default constructor, sets `m_internal` to null.

        |

        *Overload 8:*
         Constructor using raw copy of pre-existing `::fz_stext_page`.
        """
        _mupdf.FzStextPage_swiginit(self, _mupdf.new_FzStextPage(*args))
    __swig_destroy__ = _mupdf.delete_FzStextPage

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzStextPage_m_internal_value(self)
    m_internal = property(_mupdf.FzStextPage_m_internal_get, _mupdf.FzStextPage_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzStextPage_s_num_instances_get, _mupdf.FzStextPage_s_num_instances_set)