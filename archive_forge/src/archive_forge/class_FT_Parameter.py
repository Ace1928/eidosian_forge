from freetype.ft_types import *
class FT_Parameter(Structure):
    """
    A simple structure used to pass more or less generic parameters to
    FT_Open_Face.

    tag: A four-byte identification tag.

    data: A pointer to the parameter data
    """
    _fields_ = [('tag', FT_ULong), ('data', FT_Pointer)]