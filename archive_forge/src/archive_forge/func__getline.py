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
def _getline(self):
    line = self.ui.stdin.readline()
    if '' == line:
        raise EOFError
    return line.strip()