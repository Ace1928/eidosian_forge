from freetype.ft_types import *
class FT_Var_Named_Style(Structure):
    """
    A structure to model a named instance in a TrueType GX or OpenType
    variation font.
    """
    _fields_ = [('coords', POINTER(FT_Fixed)), ('strid', FT_UInt), ('psid', FT_UInt)]