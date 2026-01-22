import unicodedata
from pyglet.gl import *
from pyglet import image
class GlyphTextureAtlas(image.atlas.TextureAtlas):
    """A texture atlas containing glyphs."""
    texture_class = GlyphTexture

    def __init__(self, width=2048, height=2048, fmt=GL_RGBA, min_filter=GL_LINEAR, mag_filter=GL_LINEAR):
        self.texture = self.texture_class.create(width, height, GL_TEXTURE_2D, fmt, min_filter, mag_filter, fmt=fmt)
        self.allocator = image.atlas.Allocator(width, height)