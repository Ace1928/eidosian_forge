import bisect
from collections import defaultdict
import mmap
import os
import sys
import tempfile
import threading
from .context import reduction, assert_spawning
from . import util

        A shared memory area backed by a temporary file (POSIX).
        