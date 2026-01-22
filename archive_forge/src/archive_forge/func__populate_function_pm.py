from ctypes import c_uint, c_bool
from llvmlite.binding import ffi
from llvmlite.binding import passmanagers
def _populate_function_pm(self, pm):
    ffi.lib.LLVMPY_PassManagerBuilderPopulateFunctionPassManager(self, pm)