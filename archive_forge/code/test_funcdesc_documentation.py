import unittest
from numba import njit
from numba.core.funcdesc import PythonFunctionDescriptor, default_mangler
from numba.core.compiler import run_frontend
from numba.core.itanium_mangler import mangle_abi_tag

        This is a minimal test for the abi-tags support in the mangler.
        