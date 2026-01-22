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
def FzColorspace_fz_convert_separation_colors_outparams_fn(self, src_color, dst_seps, color_params):
    """
    Helper for out-params of class method fz_colorspace::ll_fz_convert_separation_colors() [fz_convert_separation_colors()].
    """
    dst_color = ll_fz_convert_separation_colors(self.m_internal, src_color, dst_seps.m_internal, color_params.internal())
    return dst_color