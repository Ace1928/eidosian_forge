import os
import sys
import weakref
from typing import Union, BinaryIO, Optional, Iterable
import pyglet
from pyglet.font.user import UserDefinedFontBase
from pyglet import gl
def add_directory(directory):
    """Add a directory of fonts to pyglet's search path.

    This function simply calls :meth:`pyglet.font.add_file` for each file with a ``.ttf``
    extension in the given directory. Subdirectories are not searched.

    :Parameters:
        `dir` : str
            Directory that contains font files.

    """
    for file in os.listdir(directory):
        if file[-4:].lower() == '.ttf':
            add_file(os.path.join(directory, file))