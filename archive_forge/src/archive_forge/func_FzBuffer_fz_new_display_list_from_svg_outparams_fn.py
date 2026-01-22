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
def FzBuffer_fz_new_display_list_from_svg_outparams_fn(self, base_uri, dir):
    """
    Helper for out-params of class method fz_buffer::ll_fz_new_display_list_from_svg() [fz_new_display_list_from_svg()].
    """
    ret, w, h = ll_fz_new_display_list_from_svg(self.m_internal, base_uri, dir.m_internal)
    return (FzDisplayList(ret), w, h)