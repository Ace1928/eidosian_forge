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
def PdfDocument_pdf_signature_contents_outparams_fn(self, signature):
    """
    Helper for out-params of class method pdf_document::ll_pdf_signature_contents() [pdf_signature_contents()].
    """
    ret, contents = ll_pdf_signature_contents(self.m_internal, signature.m_internal)
    return (ret, contents)