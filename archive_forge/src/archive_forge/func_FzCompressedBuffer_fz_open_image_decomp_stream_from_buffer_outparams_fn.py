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
def FzCompressedBuffer_fz_open_image_decomp_stream_from_buffer_outparams_fn(self):
    """
    Helper for out-params of class method fz_compressed_buffer::ll_fz_open_image_decomp_stream_from_buffer() [fz_open_image_decomp_stream_from_buffer()].
    """
    ret, l2factor = ll_fz_open_image_decomp_stream_from_buffer(self.m_internal)
    return (FzStream(ret), l2factor)