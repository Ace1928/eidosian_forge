import contextlib
import errno
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple
from . import debug, errors, osutils, trace
from .hooks import Hooks
from .i18n import gettext
from .transport import Transport
def _clear_f(self):
    """Clear the self.f attribute cleanly."""
    if self.f:
        self.f.close()
        self.f = None