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
Perform Compilation.

        Compilation options are accepted as keyword arguments, with the
        following considerations:

        - Underscores (`_`) in option names are converted to dashes (`-`), to
          match NVVM's option name format.
        - Options that take a value will be emitted in the form
          "-<name>=<value>".
        - Booleans passed as option values will be converted to integers.
        - Options which take no value (such as `-gen-lto`) should have a value
          of `None` passed in and will be emitted in the form "-<name>".

        For documentation on NVVM compilation options, see the CUDA Toolkit
        Documentation:

        https://docs.nvidia.com/cuda/libnvvm-api/index.html#_CPPv418nvvmCompileProgram11nvvmProgramiPPKc
        