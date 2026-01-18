import os
import sys
import runpy
import types
from . import get_start_method, set_start_method
from . import process
from .context import reduction
from . import util

    Set sys.modules['__main__'] to module at main_path
    