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
class PdfCmap(object):
    """ Wrapper class for struct `pdf_cmap`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def pdf_add_codespace(self, low, high, n):
        """ Class-aware wrapper for `::pdf_add_codespace()`."""
        return _mupdf.PdfCmap_pdf_add_codespace(self, low, high, n)

    def pdf_cmap_size(self):
        """ Class-aware wrapper for `::pdf_cmap_size()`."""
        return _mupdf.PdfCmap_pdf_cmap_size(self)

    def pdf_cmap_wmode(self):
        """ Class-aware wrapper for `::pdf_cmap_wmode()`."""
        return _mupdf.PdfCmap_pdf_cmap_wmode(self)

    def pdf_decode_cmap(self, s, e, cpt):
        """
        Class-aware wrapper for `::pdf_decode_cmap()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_decode_cmap(unsigned char *s, unsigned char *e)` => `(int, unsigned int cpt)`
        """
        return _mupdf.PdfCmap_pdf_decode_cmap(self, s, e, cpt)

    def pdf_lookup_cmap(self, cpt):
        """ Class-aware wrapper for `::pdf_lookup_cmap()`."""
        return _mupdf.PdfCmap_pdf_lookup_cmap(self, cpt)

    def pdf_lookup_cmap_full(self, cpt, out):
        """
        Class-aware wrapper for `::pdf_lookup_cmap_full()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_lookup_cmap_full(unsigned int cpt)` => `(int, int out)`
        """
        return _mupdf.PdfCmap_pdf_lookup_cmap_full(self, cpt, out)

    def pdf_map_one_to_many(self, one, many, len):
        """
        Class-aware wrapper for `::pdf_map_one_to_many()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_map_one_to_many(unsigned int one, size_t len)` => int many
        """
        return _mupdf.PdfCmap_pdf_map_one_to_many(self, one, many, len)

    def pdf_map_range_to_range(self, srclo, srchi, dstlo):
        """ Class-aware wrapper for `::pdf_map_range_to_range()`."""
        return _mupdf.PdfCmap_pdf_map_range_to_range(self, srclo, srchi, dstlo)

    def pdf_set_cmap_wmode(self, wmode):
        """ Class-aware wrapper for `::pdf_set_cmap_wmode()`."""
        return _mupdf.PdfCmap_pdf_set_cmap_wmode(self, wmode)

    def pdf_set_usecmap(self, usecmap):
        """ Class-aware wrapper for `::pdf_set_usecmap()`."""
        return _mupdf.PdfCmap_pdf_set_usecmap(self, usecmap)

    def pdf_sort_cmap(self):
        """ Class-aware wrapper for `::pdf_sort_cmap()`."""
        return _mupdf.PdfCmap_pdf_sort_cmap(self)

    def __init__(self, *args):
        """
        *Overload 1:*
        == Constructors.  Constructor using `pdf_new_cmap()`.

        |

        *Overload 2:*
        Constructor using `pdf_new_identity_cmap()`.

        |

        *Overload 3:*
        Copy constructor using `pdf_keep_cmap()`.

        |

        *Overload 4:*
        Constructor using raw copy of pre-existing `::pdf_cmap`.
        """
        _mupdf.PdfCmap_swiginit(self, _mupdf.new_PdfCmap(*args))
    __swig_destroy__ = _mupdf.delete_PdfCmap

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.PdfCmap_m_internal_value(self)
    m_internal = property(_mupdf.PdfCmap_m_internal_get, _mupdf.PdfCmap_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.PdfCmap_s_num_instances_get, _mupdf.PdfCmap_s_num_instances_set)