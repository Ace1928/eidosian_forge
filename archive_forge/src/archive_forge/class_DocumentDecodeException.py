from os.path import dirname as _dirname
from os.path import splitext as _splitext
import pyglet
from pyglet.text import layout, document, caret
class DocumentDecodeException(Exception):
    """An error occurred decoding document text."""
    pass