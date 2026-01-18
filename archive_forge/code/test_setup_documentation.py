import os
import subprocess
import sys
from distutils import version
import breezy
from .. import tests
 test cmd `python setup.py build`

        This tests that the build process and man generator run correctly.
        It also can catch new subdirectories that weren't added to setup.py.
        