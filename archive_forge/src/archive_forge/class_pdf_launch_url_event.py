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
class pdf_launch_url_event(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    url = property(_mupdf.pdf_launch_url_event_url_get, _mupdf.pdf_launch_url_event_url_set)
    new_frame = property(_mupdf.pdf_launch_url_event_new_frame_get, _mupdf.pdf_launch_url_event_new_frame_set)

    def __init__(self):
        _mupdf.pdf_launch_url_event_swiginit(self, _mupdf.new_pdf_launch_url_event())
    __swig_destroy__ = _mupdf.delete_pdf_launch_url_event