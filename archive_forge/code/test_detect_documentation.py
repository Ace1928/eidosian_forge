import os
import sys
import subprocess
import threading
from numba import cuda
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
from numba.tests.support import captured_stdout

        This tests that the find_libs works as expected in the case of an
        environment variable being used to set the path.
        