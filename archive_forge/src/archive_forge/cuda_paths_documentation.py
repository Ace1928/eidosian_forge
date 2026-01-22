import sys
import re
import os
from collections import namedtuple
from numba.core.config import IS_WIN32
from numba.misc.findlib import find_lib, find_file

    Return the Debian NVIDIA Maintainers-packaged libdevice location, if it
    exists.
    