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
def PdfDocument_pdf_lookup_page_loc_outparams_fn(self, needle):
    """
    Helper for out-params of class method pdf_document::ll_pdf_lookup_page_loc() [pdf_lookup_page_loc()].
    """
    ret, parentp, indexp = ll_pdf_lookup_page_loc(self.m_internal, needle)
    return (PdfObj(ll_pdf_keep_obj(ret)), PdfObj(ll_pdf_keep_obj(parentp)), indexp)