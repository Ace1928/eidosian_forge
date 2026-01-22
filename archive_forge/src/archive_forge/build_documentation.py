import os
import sys
from distutils.command.build import build as old_build
from distutils.util import get_platform
from numpy.distutils.command.config_compiler import show_fortran_compilers

        the '_simd' module is a very large. Adding more dispatched features
        will increase binary size and compile time. By default we minimize
        the targeted features to those most commonly used by the NumPy SIMD interface(NPYV),
        NOTE: any specified features will be ignored if they're:
            - part of the baseline(--cpu-baseline)
            - not part of dispatch-able features(--cpu-dispatch)
            - not supported by compiler or platform
        