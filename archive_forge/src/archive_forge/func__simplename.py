import os
import sys
import warnings
from os.path import basename, dirname, exists, join, splitext
from pygame.font import Font
def _simplename(name):
    """create simple version of the font name"""
    return ''.join((c.lower() for c in name if c.isalnum()))