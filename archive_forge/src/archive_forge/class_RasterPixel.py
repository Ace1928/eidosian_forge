from __future__ import annotations
import enum
class RasterPixel(enum.IntEnum):
    """Raster Type Codes."""
    Undefined = 0
    User_Defined = 32767
    IsArea = 1
    IsPoint = 2