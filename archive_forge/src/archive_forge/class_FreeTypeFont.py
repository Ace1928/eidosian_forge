from __future__ import annotations
import base64
import os
import sys
import warnings
from enum import IntEnum
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from . import Image
from ._util import is_directory, is_path
class FreeTypeFont:
    """FreeType font wrapper (requires _imagingft service)"""

    def __init__(self, font: bytes | str | Path | BinaryIO | None=None, size: float=10, index: int=0, encoding: str='', layout_engine: Layout | None=None) -> None:
        if size <= 0:
            msg = 'font size must be greater than 0'
            raise ValueError(msg)
        self.path = font
        self.size = size
        self.index = index
        self.encoding = encoding
        if layout_engine not in (Layout.BASIC, Layout.RAQM):
            layout_engine = Layout.BASIC
            if core.HAVE_RAQM:
                layout_engine = Layout.RAQM
        elif layout_engine == Layout.RAQM and (not core.HAVE_RAQM):
            warnings.warn('Raqm layout was requested, but Raqm is not available. Falling back to basic layout.')
            layout_engine = Layout.BASIC
        self.layout_engine = layout_engine

        def load_from_bytes(f):
            self.font_bytes = f.read()
            self.font = core.getfont('', size, index, encoding, self.font_bytes, layout_engine)
        if is_path(font):
            if isinstance(font, Path):
                font = str(font)
            if sys.platform == 'win32':
                font_bytes_path = font if isinstance(font, bytes) else font.encode()
                try:
                    font_bytes_path.decode('ascii')
                except UnicodeDecodeError:
                    with open(font, 'rb') as f:
                        load_from_bytes(f)
                    return
            self.font = core.getfont(font, size, index, encoding, layout_engine=layout_engine)
        else:
            load_from_bytes(font)

    def __getstate__(self):
        return [self.path, self.size, self.index, self.encoding, self.layout_engine]

    def __setstate__(self, state):
        path, size, index, encoding, layout_engine = state
        self.__init__(path, size, index, encoding, layout_engine)

    def getname(self):
        """
        :return: A tuple of the font family (e.g. Helvetica) and the font style
            (e.g. Bold)
        """
        return (self.font.family, self.font.style)

    def getmetrics(self):
        """
        :return: A tuple of the font ascent (the distance from the baseline to
            the highest outline point) and descent (the distance from the
            baseline to the lowest outline point, a negative value)
        """
        return (self.font.ascent, self.font.descent)

    def getlength(self, text, mode='', direction=None, features=None, language=None):
        """
        Returns length (in pixels with 1/64 precision) of given text when rendered
        in font with provided direction, features, and language.

        This is the amount by which following text should be offset.
        Text bounding box may extend past the length in some fonts,
        e.g. when using italics or accents.

        The result is returned as a float; it is a whole number if using basic layout.

        Note that the sum of two lengths may not equal the length of a concatenated
        string due to kerning. If you need to adjust for kerning, include the following
        character and subtract its length.

        For example, instead of ::

          hello = font.getlength("Hello")
          world = font.getlength("World")
          hello_world = hello + world  # not adjusted for kerning
          assert hello_world == font.getlength("HelloWorld")  # may fail

        use ::

          hello = font.getlength("HelloW") - font.getlength("W")  # adjusted for kerning
          world = font.getlength("World")
          hello_world = hello + world  # adjusted for kerning
          assert hello_world == font.getlength("HelloWorld")  # True

        or disable kerning with (requires libraqm) ::

          hello = draw.textlength("Hello", font, features=["-kern"])
          world = draw.textlength("World", font, features=["-kern"])
          hello_world = hello + world  # kerning is disabled, no need to adjust
          assert hello_world == draw.textlength("HelloWorld", font, features=["-kern"])

        .. versionadded:: 8.0.0

        :param text: Text to measure.
        :param mode: Used by some graphics drivers to indicate what mode the
                     driver prefers; if empty, the renderer may return either
                     mode. Note that the mode is always a string, to simplify
                     C-level implementations.

        :param direction: Direction of the text. It can be 'rtl' (right to
                          left), 'ltr' (left to right) or 'ttb' (top to bottom).
                          Requires libraqm.

        :param features: A list of OpenType font features to be used during text
                         layout. This is usually used to turn on optional
                         font features that are not enabled by default,
                         for example 'dlig' or 'ss01', but can be also
                         used to turn off default font features for
                         example '-liga' to disable ligatures or '-kern'
                         to disable kerning.  To get all supported
                         features, see
                         https://learn.microsoft.com/en-us/typography/opentype/spec/featurelist
                         Requires libraqm.

        :param language: Language of the text. Different languages may use
                         different glyph shapes or ligatures. This parameter tells
                         the font which language the text is in, and to apply the
                         correct substitutions as appropriate, if available.
                         It should be a `BCP 47 language code
                         <https://www.w3.org/International/articles/language-tags/>`_
                         Requires libraqm.

        :return: Either width for horizontal text, or height for vertical text.
        """
        _string_length_check(text)
        return self.font.getlength(text, mode, direction, features, language) / 64

    def getbbox(self, text, mode='', direction=None, features=None, language=None, stroke_width=0, anchor=None):
        """
        Returns bounding box (in pixels) of given text relative to given anchor
        when rendered in font with provided direction, features, and language.

        Use :py:meth:`getlength()` to get the offset of following text with
        1/64 pixel precision. The bounding box includes extra margins for
        some fonts, e.g. italics or accents.

        .. versionadded:: 8.0.0

        :param text: Text to render.
        :param mode: Used by some graphics drivers to indicate what mode the
                     driver prefers; if empty, the renderer may return either
                     mode. Note that the mode is always a string, to simplify
                     C-level implementations.

        :param direction: Direction of the text. It can be 'rtl' (right to
                          left), 'ltr' (left to right) or 'ttb' (top to bottom).
                          Requires libraqm.

        :param features: A list of OpenType font features to be used during text
                         layout. This is usually used to turn on optional
                         font features that are not enabled by default,
                         for example 'dlig' or 'ss01', but can be also
                         used to turn off default font features for
                         example '-liga' to disable ligatures or '-kern'
                         to disable kerning.  To get all supported
                         features, see
                         https://learn.microsoft.com/en-us/typography/opentype/spec/featurelist
                         Requires libraqm.

        :param language: Language of the text. Different languages may use
                         different glyph shapes or ligatures. This parameter tells
                         the font which language the text is in, and to apply the
                         correct substitutions as appropriate, if available.
                         It should be a `BCP 47 language code
                         <https://www.w3.org/International/articles/language-tags/>`_
                         Requires libraqm.

        :param stroke_width: The width of the text stroke.

        :param anchor:  The text anchor alignment. Determines the relative location of
                        the anchor to the text. The default alignment is top left,
                        specifically ``la`` for horizontal text and ``lt`` for
                        vertical text. See :ref:`text-anchors` for details.

        :return: ``(left, top, right, bottom)`` bounding box
        """
        _string_length_check(text)
        size, offset = self.font.getsize(text, mode, direction, features, language, anchor)
        left, top = (offset[0] - stroke_width, offset[1] - stroke_width)
        width, height = (size[0] + 2 * stroke_width, size[1] + 2 * stroke_width)
        return (left, top, left + width, top + height)

    def getmask(self, text, mode='', direction=None, features=None, language=None, stroke_width=0, anchor=None, ink=0, start=None):
        """
        Create a bitmap for the text.

        If the font uses antialiasing, the bitmap should have mode ``L`` and use a
        maximum value of 255. If the font has embedded color data, the bitmap
        should have mode ``RGBA``. Otherwise, it should have mode ``1``.

        :param text: Text to render.
        :param mode: Used by some graphics drivers to indicate what mode the
                     driver prefers; if empty, the renderer may return either
                     mode. Note that the mode is always a string, to simplify
                     C-level implementations.

                     .. versionadded:: 1.1.5

        :param direction: Direction of the text. It can be 'rtl' (right to
                          left), 'ltr' (left to right) or 'ttb' (top to bottom).
                          Requires libraqm.

                          .. versionadded:: 4.2.0

        :param features: A list of OpenType font features to be used during text
                         layout. This is usually used to turn on optional
                         font features that are not enabled by default,
                         for example 'dlig' or 'ss01', but can be also
                         used to turn off default font features for
                         example '-liga' to disable ligatures or '-kern'
                         to disable kerning.  To get all supported
                         features, see
                         https://learn.microsoft.com/en-us/typography/opentype/spec/featurelist
                         Requires libraqm.

                         .. versionadded:: 4.2.0

        :param language: Language of the text. Different languages may use
                         different glyph shapes or ligatures. This parameter tells
                         the font which language the text is in, and to apply the
                         correct substitutions as appropriate, if available.
                         It should be a `BCP 47 language code
                         <https://www.w3.org/International/articles/language-tags/>`_
                         Requires libraqm.

                         .. versionadded:: 6.0.0

        :param stroke_width: The width of the text stroke.

                         .. versionadded:: 6.2.0

        :param anchor:  The text anchor alignment. Determines the relative location of
                        the anchor to the text. The default alignment is top left,
                        specifically ``la`` for horizontal text and ``lt`` for
                        vertical text. See :ref:`text-anchors` for details.

                         .. versionadded:: 8.0.0

        :param ink: Foreground ink for rendering in RGBA mode.

                         .. versionadded:: 8.0.0

        :param start: Tuple of horizontal and vertical offset, as text may render
                      differently when starting at fractional coordinates.

                         .. versionadded:: 9.4.0

        :return: An internal PIL storage memory instance as defined by the
                 :py:mod:`PIL.Image.core` interface module.
        """
        return self.getmask2(text, mode, direction=direction, features=features, language=language, stroke_width=stroke_width, anchor=anchor, ink=ink, start=start)[0]

    def getmask2(self, text, mode='', direction=None, features=None, language=None, stroke_width=0, anchor=None, ink=0, start=None, *args, **kwargs):
        """
        Create a bitmap for the text.

        If the font uses antialiasing, the bitmap should have mode ``L`` and use a
        maximum value of 255. If the font has embedded color data, the bitmap
        should have mode ``RGBA``. Otherwise, it should have mode ``1``.

        :param text: Text to render.
        :param mode: Used by some graphics drivers to indicate what mode the
                     driver prefers; if empty, the renderer may return either
                     mode. Note that the mode is always a string, to simplify
                     C-level implementations.

                     .. versionadded:: 1.1.5

        :param direction: Direction of the text. It can be 'rtl' (right to
                          left), 'ltr' (left to right) or 'ttb' (top to bottom).
                          Requires libraqm.

                          .. versionadded:: 4.2.0

        :param features: A list of OpenType font features to be used during text
                         layout. This is usually used to turn on optional
                         font features that are not enabled by default,
                         for example 'dlig' or 'ss01', but can be also
                         used to turn off default font features for
                         example '-liga' to disable ligatures or '-kern'
                         to disable kerning.  To get all supported
                         features, see
                         https://learn.microsoft.com/en-us/typography/opentype/spec/featurelist
                         Requires libraqm.

                         .. versionadded:: 4.2.0

        :param language: Language of the text. Different languages may use
                         different glyph shapes or ligatures. This parameter tells
                         the font which language the text is in, and to apply the
                         correct substitutions as appropriate, if available.
                         It should be a `BCP 47 language code
                         <https://www.w3.org/International/articles/language-tags/>`_
                         Requires libraqm.

                         .. versionadded:: 6.0.0

        :param stroke_width: The width of the text stroke.

                         .. versionadded:: 6.2.0

        :param anchor:  The text anchor alignment. Determines the relative location of
                        the anchor to the text. The default alignment is top left,
                        specifically ``la`` for horizontal text and ``lt`` for
                        vertical text. See :ref:`text-anchors` for details.

                         .. versionadded:: 8.0.0

        :param ink: Foreground ink for rendering in RGBA mode.

                         .. versionadded:: 8.0.0

        :param start: Tuple of horizontal and vertical offset, as text may render
                      differently when starting at fractional coordinates.

                         .. versionadded:: 9.4.0

        :return: A tuple of an internal PIL storage memory instance as defined by the
                 :py:mod:`PIL.Image.core` interface module, and the text offset, the
                 gap between the starting coordinate and the first marking
        """
        _string_length_check(text)
        if start is None:
            start = (0, 0)
        im = None
        size = None

        def fill(width, height):
            nonlocal im, size
            size = (width, height)
            if Image.MAX_IMAGE_PIXELS is not None:
                pixels = max(1, width) * max(1, height)
                if pixels > 2 * Image.MAX_IMAGE_PIXELS:
                    return
            im = Image.core.fill('RGBA' if mode == 'RGBA' else 'L', size)
            return im
        offset = self.font.render(text, fill, mode, direction, features, language, stroke_width, anchor, ink, start[0], start[1])
        Image._decompression_bomb_check(size)
        return (im, offset)

    def font_variant(self, font=None, size=None, index=None, encoding=None, layout_engine=None):
        """
        Create a copy of this FreeTypeFont object,
        using any specified arguments to override the settings.

        Parameters are identical to the parameters used to initialize this
        object.

        :return: A FreeTypeFont object.
        """
        if font is None:
            try:
                font = BytesIO(self.font_bytes)
            except AttributeError:
                font = self.path
        return FreeTypeFont(font=font, size=self.size if size is None else size, index=self.index if index is None else index, encoding=self.encoding if encoding is None else encoding, layout_engine=layout_engine or self.layout_engine)

    def get_variation_names(self):
        """
        :returns: A list of the named styles in a variation font.
        :exception OSError: If the font is not a variation font.
        """
        try:
            names = self.font.getvarnames()
        except AttributeError as e:
            msg = 'FreeType 2.9.1 or greater is required'
            raise NotImplementedError(msg) from e
        return [name.replace(b'\x00', b'') for name in names]

    def set_variation_by_name(self, name):
        """
        :param name: The name of the style.
        :exception OSError: If the font is not a variation font.
        """
        names = self.get_variation_names()
        if not isinstance(name, bytes):
            name = name.encode()
        index = names.index(name) + 1
        if index == getattr(self, '_last_variation_index', None):
            return
        self._last_variation_index = index
        self.font.setvarname(index)

    def get_variation_axes(self):
        """
        :returns: A list of the axes in a variation font.
        :exception OSError: If the font is not a variation font.
        """
        try:
            axes = self.font.getvaraxes()
        except AttributeError as e:
            msg = 'FreeType 2.9.1 or greater is required'
            raise NotImplementedError(msg) from e
        for axis in axes:
            axis['name'] = axis['name'].replace(b'\x00', b'')
        return axes

    def set_variation_by_axes(self, axes):
        """
        :param axes: A list of values for each axis.
        :exception OSError: If the font is not a variation font.
        """
        try:
            self.font.setvaraxes(axes)
        except AttributeError as e:
            msg = 'FreeType 2.9.1 or greater is required'
            raise NotImplementedError(msg) from e