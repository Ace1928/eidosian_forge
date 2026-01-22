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
def FzFunction_fz_eval_function_outparams_fn(self, in_, inlen, outlen):
    """
    Helper for out-params of class method fz_function::ll_fz_eval_function() [fz_eval_function()].
    """
    out = ll_fz_eval_function(self.m_internal, in_, inlen, outlen)
    return out