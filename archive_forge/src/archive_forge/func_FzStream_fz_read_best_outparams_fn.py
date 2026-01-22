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
def FzStream_fz_read_best_outparams_fn(self, initial, worst_case):
    """
    Helper for out-params of class method fz_stream::ll_fz_read_best() [fz_read_best()].
    """
    ret, truncated = ll_fz_read_best(self.m_internal, initial, worst_case)
    return (FzBuffer(ret), truncated)