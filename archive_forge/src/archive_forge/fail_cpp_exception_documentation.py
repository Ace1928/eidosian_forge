import sys
import greenlet
from greenlet.tests import _test_extension_cpp

Helper for testing a C++ exception throw aborts the process.

Takes one argument, the name of the function in :mod:`_test_extension_cpp` to call.
