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
def FzDocument_fz_resolve_link_outparams_fn(self, uri):
    """
    Helper for out-params of class method fz_document::ll_fz_resolve_link() [fz_resolve_link()].
    """
    ret, xp, yp = ll_fz_resolve_link(self.m_internal, uri)
    return (FzLocation(ret), xp, yp)