import os
from distutils.msvccompiler import MSVCCompiler as _MSVCCompiler
from .system_info import platform_bits
 Add flags if we are using MSVC compiler

    We can't see `build_cmd` in our scope, because we have not initialized
    the distutils build command, so use this deferred calculation to run
    when we are building the library.
    