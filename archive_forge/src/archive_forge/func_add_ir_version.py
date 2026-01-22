import logging
import re
import sys
import warnings
from ctypes import (c_void_p, c_int, POINTER, c_char_p, c_size_t, byref,
import threading
from llvmlite import ir
from .error import NvvmError, NvvmSupportError, NvvmWarning
from .libs import get_libdevice, open_libdevice, open_cudalib
from numba.core import cgutils, config
def add_ir_version(mod):
    """Add NVVM IR version to module"""
    i32 = ir.IntType(32)
    ir_versions = [i32(v) for v in NVVM().get_ir_version()]
    md_ver = mod.add_metadata(ir_versions)
    mod.add_named_metadata('nvvmir.version', md_ver)