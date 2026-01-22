import codecs
import io
import os
import sys
import warnings
from ..lazy_import import lazy_import
import time
from breezy import (
from .. import config, osutils, trace
from . import NullProgressView, UIFactory
def _avail_width(self):
    w = osutils.terminal_width()
    if w is None:
        return None
    else:
        return w - 1