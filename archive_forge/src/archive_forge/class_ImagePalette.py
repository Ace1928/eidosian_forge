from __future__ import annotations
import array
from . import GimpGradientFile, GimpPaletteFile, ImageColor, PaletteFile
class ImagePalette:
    """
    Color palette for palette mapped images

    :param mode: The mode to use for the palette. See:
        :ref:`concept-modes`. Defaults to "RGB"
    :param palette: An optional palette. If given, it must be a bytearray,
        an array or a list of ints between 0-255. The list must consist of
        all channels for one color followed by the next color (e.g. RGBRGBRGB).
        Defaults to an empty palette.
    """

    def __init__(self, mode='RGB', palette=None):
        self.mode = mode
        self.rawmode = None
        self.palette = palette or bytearray()
        self.dirty = None

    @property
    def palette(self):
        return self._palette

    @palette.setter
    def palette(self, palette):
        self._colors = None
        self._palette = palette

    @property
    def colors(self):
        if self._colors is None:
            mode_len = len(self.mode)
            self._colors = {}
            for i in range(0, len(self.palette), mode_len):
                color = tuple(self.palette[i:i + mode_len])
                if color in self._colors:
                    continue
                self._colors[color] = i // mode_len
        return self._colors

    @colors.setter
    def colors(self, colors):
        self._colors = colors

    def copy(self):
        new = ImagePalette()
        new.mode = self.mode
        new.rawmode = self.rawmode
        if self.palette is not None:
            new.palette = self.palette[:]
        new.dirty = self.dirty
        return new

    def getdata(self):
        """
        Get palette contents in format suitable for the low-level
        ``im.putpalette`` primitive.

        .. warning:: This method is experimental.
        """
        if self.rawmode:
            return (self.rawmode, self.palette)
        return (self.mode, self.tobytes())

    def tobytes(self):
        """Convert palette to bytes.

        .. warning:: This method is experimental.
        """
        if self.rawmode:
            msg = 'palette contains raw palette data'
            raise ValueError(msg)
        if isinstance(self.palette, bytes):
            return self.palette
        arr = array.array('B', self.palette)
        return arr.tobytes()
    tostring = tobytes

    def _new_color_index(self, image=None, e=None):
        if not isinstance(self.palette, bytearray):
            self._palette = bytearray(self.palette)
        index = len(self.palette) // 3
        special_colors = ()
        if image:
            special_colors = (image.info.get('background'), image.info.get('transparency'))
            while index in special_colors:
                index += 1
        if index >= 256:
            if image:
                for i, count in reversed(list(enumerate(image.histogram()))):
                    if count == 0 and i not in special_colors:
                        index = i
                        break
            if index >= 256:
                msg = 'cannot allocate more than 256 colors'
                raise ValueError(msg) from e
        return index

    def getcolor(self, color, image=None):
        """Given an rgb tuple, allocate palette entry.

        .. warning:: This method is experimental.
        """
        if self.rawmode:
            msg = 'palette contains raw palette data'
            raise ValueError(msg)
        if isinstance(color, tuple):
            if self.mode == 'RGB':
                if len(color) == 4:
                    if color[3] != 255:
                        msg = 'cannot add non-opaque RGBA color to RGB palette'
                        raise ValueError(msg)
                    color = color[:3]
            elif self.mode == 'RGBA':
                if len(color) == 3:
                    color += (255,)
            try:
                return self.colors[color]
            except KeyError as e:
                index = self._new_color_index(image, e)
                self.colors[color] = index
                if index * 3 < len(self.palette):
                    self._palette = self.palette[:index * 3] + bytes(color) + self.palette[index * 3 + 3:]
                else:
                    self._palette += bytes(color)
                self.dirty = 1
                return index
        else:
            msg = f'unknown color specifier: {repr(color)}'
            raise ValueError(msg)

    def save(self, fp):
        """Save palette to text file.

        .. warning:: This method is experimental.
        """
        if self.rawmode:
            msg = 'palette contains raw palette data'
            raise ValueError(msg)
        if isinstance(fp, str):
            fp = open(fp, 'w')
        fp.write('# Palette\n')
        fp.write(f'# Mode: {self.mode}\n')
        for i in range(256):
            fp.write(f'{i}')
            for j in range(i * len(self.mode), (i + 1) * len(self.mode)):
                try:
                    fp.write(f' {self.palette[j]}')
                except IndexError:
                    fp.write(' 0')
            fp.write('\n')
        fp.close()