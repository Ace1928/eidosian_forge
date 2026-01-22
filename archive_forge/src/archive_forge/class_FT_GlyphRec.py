from freetype.ft_types import *
class FT_GlyphRec(Structure):
    """
    The root glyph structure contains a given glyph image plus its advance
    width in 16.16 fixed float format.

    library:  A handle to the FreeType library object.

    clazz: A pointer to the glyph's class. Private.

    format: The format of the glyph's image.

    advance: A 16.16 vector that gives the glyph's advance width.
    """
    _fields_ = [('library', FT_Library), ('clazz', c_void_p), ('format', FT_Glyph_Format), ('advance', FT_Vector)]