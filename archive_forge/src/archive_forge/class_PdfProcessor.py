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
class PdfProcessor(object):
    """ Wrapper class for struct `pdf_processor`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    @staticmethod
    def pdf_new_color_filter(doc, chain, struct_parents, transform, options, copts):
        """ Class-aware wrapper for `::pdf_new_color_filter()`."""
        return _mupdf.PdfProcessor_pdf_new_color_filter(doc, chain, struct_parents, transform, options, copts)

    def pdf_close_processor(self):
        """ Class-aware wrapper for `::pdf_close_processor()`."""
        return _mupdf.PdfProcessor_pdf_close_processor(self)

    def pdf_process_annot(self, annot, cookie):
        """ Class-aware wrapper for `::pdf_process_annot()`."""
        return _mupdf.PdfProcessor_pdf_process_annot(self, annot, cookie)

    def pdf_process_contents(self, doc, res, stm, cookie, out_res):
        """
        Class-aware wrapper for `::pdf_process_contents()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_process_contents(::pdf_document *doc, ::pdf_obj *res, ::pdf_obj *stm, ::fz_cookie *cookie, ::pdf_obj **out_res)` =>
        """
        return _mupdf.PdfProcessor_pdf_process_contents(self, doc, res, stm, cookie, out_res)

    def pdf_process_glyph(self, doc, resources, contents):
        """ Class-aware wrapper for `::pdf_process_glyph()`."""
        return _mupdf.PdfProcessor_pdf_process_glyph(self, doc, resources, contents)

    def pdf_process_raw_contents(self, doc, rdb, stmobj, cookie):
        """ Class-aware wrapper for `::pdf_process_raw_contents()`."""
        return _mupdf.PdfProcessor_pdf_process_raw_contents(self, doc, rdb, stmobj, cookie)

    def pdf_processor_pop_resources(self):
        """ Class-aware wrapper for `::pdf_processor_pop_resources()`."""
        return _mupdf.PdfProcessor_pdf_processor_pop_resources(self)

    def pdf_processor_push_resources(self, res):
        """ Class-aware wrapper for `::pdf_processor_push_resources()`."""
        return _mupdf.PdfProcessor_pdf_processor_push_resources(self, res)

    def __init__(self, *args):
        """
        *Overload 1:*
        == Constructors.  Constructor using `pdf_new_buffer_processor()`.

        |

        *Overload 2:*
        Constructor using `pdf_new_output_processor()`.

        |

        *Overload 3:*
        Constructor using `pdf_new_run_processor()`.

        |

        *Overload 4:*
        Constructor using `pdf_new_sanitize_filter()`.

        |

        *Overload 5:*
        Copy constructor using `pdf_keep_processor()`.

        |

        *Overload 6:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 7:*
        Constructor using raw copy of pre-existing `::pdf_processor`.
        """
        _mupdf.PdfProcessor_swiginit(self, _mupdf.new_PdfProcessor(*args))
    __swig_destroy__ = _mupdf.delete_PdfProcessor

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.PdfProcessor_m_internal_value(self)
    m_internal = property(_mupdf.PdfProcessor_m_internal_get, _mupdf.PdfProcessor_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.PdfProcessor_s_num_instances_get, _mupdf.PdfProcessor_s_num_instances_set)