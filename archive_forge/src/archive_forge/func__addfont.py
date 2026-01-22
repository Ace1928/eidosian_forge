import os
import sys
import warnings
from os.path import basename, dirname, exists, join, splitext
from pygame.font import Font
def _addfont(name, bold, italic, font, fontdict):
    """insert a font and style into the font dictionary"""
    if name not in fontdict:
        fontdict[name] = {}
    fontdict[name][bold, italic] = font