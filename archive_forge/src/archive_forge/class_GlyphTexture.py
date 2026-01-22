import unicodedata
from pyglet.gl import *
from pyglet import image
class GlyphTexture(image.Texture):
    region_class = Glyph