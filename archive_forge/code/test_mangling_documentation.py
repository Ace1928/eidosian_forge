from numba.core import types
from numba.core.funcdesc import default_mangler
from numba.tests.support import unittest, TestCase

Test function name mangling.
The mangling affects the ABI of numba compiled binaries.
