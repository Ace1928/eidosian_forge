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
def FzStream_fz_decomp_image_from_stream_outparams_fn(self, image, subarea, indexed, l2factor):
    """
    Helper for out-params of class method fz_stream::ll_fz_decomp_image_from_stream() [fz_decomp_image_from_stream()].
    """
    ret, l2extra = ll_fz_decomp_image_from_stream(self.m_internal, image.m_internal, subarea.internal(), indexed, l2factor)
    return (FzPixmap(ret), l2extra)