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
def FzPixmap_fz_fill_pixmap_with_color_outparams_fn(self, colorspace, color_params):
    """
    Helper for out-params of class method fz_pixmap::ll_fz_fill_pixmap_with_color() [fz_fill_pixmap_with_color()].
    """
    color = ll_fz_fill_pixmap_with_color(self.m_internal, colorspace.m_internal, color_params.internal())
    return color