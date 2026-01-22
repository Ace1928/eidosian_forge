from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Protocol, List
import pyglet
from pyglet.font import base
class UserDefinedMappingFont(UserDefinedFontBase):
    """The default UserDefinedFont, it can take mappings of characters to ImageData to make a User defined font."""

    def __init__(self, name: str, default_char: str, size: int, mappings: DictLikeObject, ascent: Optional[int]=None, descent: Optional[int]=None, bold: bool=False, italic: bool=False, stretch: bool=False, dpi: int=96, locale: Optional[str]=None):
        """Create a custom font using the mapping dict.

        :Parameters:
            `name` : str
                Name of the font.
            `default_char` : str
                If a character in a string is not found in the font,
                it will use this as fallback.
            `size` : int
                Font size.
            `mappings` : DictLikeObject
                A dict or dict-like object with a get function.
                The get function must take a string character, and output ImageData if found.
                It also must return None if no character is found.
            `ascent` : int
                Maximum ascent above the baseline, in pixels. If None, the image height is used.
            `descent` : int
                Maximum descent below the baseline, in pixels. Usually negative.
        """
        self.mappings = mappings
        default_image = self.mappings.get(default_char)
        if not default_image:
            raise UserDefinedFontException(f"Default character '{default_char}' must exist within your mappings.")
        if ascent is None or descent is None:
            if ascent is None:
                ascent = default_image.height
            if descent is None:
                descent = 0
        super().__init__(name, default_char, size, ascent, descent, bold, italic, stretch, dpi, locale)

    def enable_scaling(self, base_size: int) -> None:
        super().enable_scaling(base_size)
        glyphs = self.get_glyphs(self.default_char)
        self.ascent = glyphs[0].height
        self.descent = 0

    def get_glyphs(self, text: str) -> List[Glyph]:
        """Create and return a list of Glyphs for `text`.

        If any characters do not have a known glyph representation in this
        font, a substitution will be made with the default_char.

        :Parameters:
            `text` : str or unicode
                Text to render.

        :rtype: list of `Glyph`
        """
        glyph_renderer = None
        glyphs = []
        for c in base.get_grapheme_clusters(text):
            if c == '\t':
                c = ' '
            if c not in self.glyphs:
                if not glyph_renderer:
                    glyph_renderer = self.glyph_renderer_class(self)
                image_data = self.mappings.get(c)
                if not image_data:
                    c = self.default_char
                else:
                    self.glyphs[c] = glyph_renderer.render(image_data)
            glyphs.append(self.glyphs[c])
        return glyphs