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
def PdfDocument_pdf_load_to_unicode_outparams_fn(self, font, collection, cmapstm):
    """
    Helper for out-params of class method pdf_document::ll_pdf_load_to_unicode() [pdf_load_to_unicode()].
    """
    strings = ll_pdf_load_to_unicode(self.m_internal, font.m_internal, collection, cmapstm.m_internal)
    return strings