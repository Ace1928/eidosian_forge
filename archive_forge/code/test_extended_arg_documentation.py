import unittest
import dis
import struct
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase

        Get a function with a EXTENDED_ARG opcode before a LOAD_CONST opcode.
        