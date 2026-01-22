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
class FzDocumentWriter(object):
    """ Wrapper class for struct `fz_document_writer`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    PathType_CBZ = _mupdf.FzDocumentWriter_PathType_CBZ
    PathType_DOCX = _mupdf.FzDocumentWriter_PathType_DOCX
    PathType_ODT = _mupdf.FzDocumentWriter_PathType_ODT
    PathType_PAM_PIXMAP = _mupdf.FzDocumentWriter_PathType_PAM_PIXMAP
    PathType_PBM_PIXMAP = _mupdf.FzDocumentWriter_PathType_PBM_PIXMAP
    PathType_PCL = _mupdf.FzDocumentWriter_PathType_PCL
    PathType_PCLM = _mupdf.FzDocumentWriter_PathType_PCLM
    PathType_PDF = _mupdf.FzDocumentWriter_PathType_PDF
    PathType_PDFOCR = _mupdf.FzDocumentWriter_PathType_PDFOCR
    PathType_PGM_PIXMAP = _mupdf.FzDocumentWriter_PathType_PGM_PIXMAP
    PathType_PKM_PIXMAP = _mupdf.FzDocumentWriter_PathType_PKM_PIXMAP
    PathType_PNG_PIXMAP = _mupdf.FzDocumentWriter_PathType_PNG_PIXMAP
    PathType_PNM_PIXMAP = _mupdf.FzDocumentWriter_PathType_PNM_PIXMAP
    PathType_PPM_PIXMAP = _mupdf.FzDocumentWriter_PathType_PPM_PIXMAP
    PathType_PS = _mupdf.FzDocumentWriter_PathType_PS
    PathType_PWG = _mupdf.FzDocumentWriter_PathType_PWG
    PathType_SVG = _mupdf.FzDocumentWriter_PathType_SVG
    OutputType_CBZ = _mupdf.FzDocumentWriter_OutputType_CBZ
    OutputType_DOCX = _mupdf.FzDocumentWriter_OutputType_DOCX
    OutputType_ODT = _mupdf.FzDocumentWriter_OutputType_ODT
    OutputType_PCL = _mupdf.FzDocumentWriter_OutputType_PCL
    OutputType_PCLM = _mupdf.FzDocumentWriter_OutputType_PCLM
    OutputType_PDF = _mupdf.FzDocumentWriter_OutputType_PDF
    OutputType_PDFOCR = _mupdf.FzDocumentWriter_OutputType_PDFOCR
    OutputType_PS = _mupdf.FzDocumentWriter_OutputType_PS
    OutputType_PWG = _mupdf.FzDocumentWriter_OutputType_PWG
    FormatPathType_DOCUMENT = _mupdf.FzDocumentWriter_FormatPathType_DOCUMENT
    FormatPathType_TEXT = _mupdf.FzDocumentWriter_FormatPathType_TEXT

    def fz_begin_page(self, mediabox):
        """
        Class-aware wrapper for `::fz_begin_page()`.
        	Called to start the process of writing a page to
        	a document.

        	mediabox: page size rectangle in points.

        	Returns a borrowed fz_device to write page contents to. This
        	should be kept if required, and only dropped if it was kept.
        """
        return _mupdf.FzDocumentWriter_fz_begin_page(self, mediabox)

    def fz_close_document_writer(self):
        """
        Class-aware wrapper for `::fz_close_document_writer()`.
        	Called to end the process of writing
        	pages to a document.

        	This writes any file level trailers required. After this
        	completes successfully the file is up to date and complete.
        """
        return _mupdf.FzDocumentWriter_fz_close_document_writer(self)

    def fz_end_page(self):
        """
        Class-aware wrapper for `::fz_end_page()`.
        	Called to end the process of writing a page to a
        	document.
        """
        return _mupdf.FzDocumentWriter_fz_end_page(self)

    def fz_pdfocr_writer_set_progress(self, progress, arg_2):
        """ Class-aware wrapper for `::fz_pdfocr_writer_set_progress()`."""
        return _mupdf.FzDocumentWriter_fz_pdfocr_writer_set_progress(self, progress, arg_2)

    def fz_write_document(self, doc):
        """
        Class-aware wrapper for `::fz_write_document()`.
        	Convenience function to feed all the pages of a document to
        	fz_begin_page/fz_run_page/fz_end_page.
        """
        return _mupdf.FzDocumentWriter_fz_write_document(self, doc)

    def fz_write_stabilized_story(self, user_css, em, contentfn, contentfn_ref, rectfn, rectfn_ref, pagefn, pagefn_ref, dir):
        """ Class-aware wrapper for `::fz_write_stabilized_story()`."""
        return _mupdf.FzDocumentWriter_fz_write_stabilized_story(self, user_css, em, contentfn, contentfn_ref, rectfn, rectfn_ref, pagefn, pagefn_ref, dir)

    def fz_write_story(self, story, rectfn, rectfn_ref, positionfn, positionfn_ref, pagefn, pagefn_ref):
        """ Class-aware wrapper for `::fz_write_story()`."""
        return _mupdf.FzDocumentWriter_fz_write_story(self, story, rectfn, rectfn_ref, positionfn, positionfn_ref, pagefn, pagefn_ref)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_document_writer_of_size()`.
        		Internal function to allocate a
        		block for a derived document_writer structure, with the base
        		structure's function pointers populated correctly, and the extra
        		space zero initialised.


        |

        *Overload 2:*
         Constructor using `fz_new_document_writer_with_buffer()`.

        |

        *Overload 3:*
         Constructor using `fz_new_jpeg_pixmap_writer()`.

        |

        *Overload 4:*
         Constructor using `fz_new_pixmap_writer()`.

        |

        *Overload 5:*
         Constructor using one of:
        		fz_new_cbz_writer()
        		fz_new_docx_writer()
        		fz_new_odt_writer()
        		fz_new_pam_pixmap_writer()
        		fz_new_pbm_pixmap_writer()
        		fz_new_pcl_writer()
        		fz_new_pclm_writer()
        		fz_new_pdf_writer()
        		fz_new_pdfocr_writer()
        		fz_new_pgm_pixmap_writer()
        		fz_new_pkm_pixmap_writer()
        		fz_new_png_pixmap_writer()
        		fz_new_pnm_pixmap_writer()
        		fz_new_ppm_pixmap_writer()
        		fz_new_ps_writer()
        		fz_new_pwg_writer()
        		fz_new_svg_writer()


        |

        *Overload 6:*
         Constructor using one of:
        		fz_new_cbz_writer_with_output()
        		fz_new_docx_writer_with_output()
        		fz_new_odt_writer_with_output()
        		fz_new_pcl_writer_with_output()
        		fz_new_pclm_writer_with_output()
        		fz_new_pdf_writer_with_output()
        		fz_new_pdfocr_writer_with_output()
        		fz_new_ps_writer_with_output()
        		fz_new_pwg_writer_with_output()

        	This constructor takes ownership of <out> -
        	out.m_internal is set to NULL after this constructor
        	returns so <out> must not be used again.


        |

        *Overload 7:*
         Constructor using one of:
        		fz_new_document_writer()
        		fz_new_text_writer()


        |

        *Overload 8:*
         Constructor using fz_new_document_writer_with_output().

        	This constructor takes ownership of <out> -
        	out.m_internal is set to NULL after this constructor
        	returns so <out> must not be used again.


        |

        *Overload 9:*
         Constructor using fz_new_text_writer_with_output().

        	This constructor takes ownership of <out> -
        	out.m_internal is set to NULL after this constructor
        	returns so <out> must not be used again.


        |

        *Overload 10:*
         Default constructor, sets `m_internal` to null.

        |

        *Overload 11:*
         Constructor using raw copy of pre-existing `::fz_document_writer`.
        """
        _mupdf.FzDocumentWriter_swiginit(self, _mupdf.new_FzDocumentWriter(*args))
    __swig_destroy__ = _mupdf.delete_FzDocumentWriter

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzDocumentWriter_m_internal_value(self)
    m_internal = property(_mupdf.FzDocumentWriter_m_internal_get, _mupdf.FzDocumentWriter_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzDocumentWriter_s_num_instances_get, _mupdf.FzDocumentWriter_s_num_instances_set)