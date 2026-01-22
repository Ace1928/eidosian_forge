import os
import platform
import sys
import warnings
from abc import ABC, abstractmethod
from pygame import error
def _colorspace_not_available(*args):
    raise RuntimeError('pygame is not built with colorspace support')