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
class fz_device_container_stack(object):
    """
    The device structure is public to allow devices to be
    implemented outside of fitz.

    Device methods should always be called using e.g.
    fz_fill_path(ctx, dev, ...) rather than
    dev->fill_path(ctx, dev, ...)

    Devices can keep track of containers (clips/masks/groups/tiles)
    as they go to save callers having to do it.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    scissor = property(_mupdf.fz_device_container_stack_scissor_get, _mupdf.fz_device_container_stack_scissor_set)
    type = property(_mupdf.fz_device_container_stack_type_get, _mupdf.fz_device_container_stack_type_set)
    user = property(_mupdf.fz_device_container_stack_user_get, _mupdf.fz_device_container_stack_user_set)

    def __init__(self):
        _mupdf.fz_device_container_stack_swiginit(self, _mupdf.new_fz_device_container_stack())
    __swig_destroy__ = _mupdf.delete_fz_device_container_stack