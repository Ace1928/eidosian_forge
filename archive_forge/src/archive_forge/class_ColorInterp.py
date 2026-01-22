from enum import Enum, IntEnum
class ColorInterp(IntEnum):
    """Raster band color interpretation."""
    undefined = 0
    gray = 1
    grey = 1
    palette = 2
    red = 3
    green = 4
    blue = 5
    alpha = 6
    hue = 7
    saturation = 8
    lightness = 9
    cyan = 10
    magenta = 11
    yellow = 12
    black = 13
    Y = 14
    Cb = 15
    Cr = 16