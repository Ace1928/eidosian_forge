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
def PdfObj_pdf_walk_tree_outparams_fn(self, arrive, leave, arg):
    """
    Helper for out-params of class method pdf_obj::ll_pdf_walk_tree() [pdf_walk_tree()].
    """
    names, values = ll_pdf_walk_tree(self.m_internal, arrive, leave, arg)
    return (PdfObj(ll_pdf_keep_obj(names)), PdfObj(ll_pdf_keep_obj(values)))