import sys
import traceback
from mako import compat
from mako import util
def _install_highlighting():
    try:
        _install_pygments()
    except ImportError:
        _install_fallback()