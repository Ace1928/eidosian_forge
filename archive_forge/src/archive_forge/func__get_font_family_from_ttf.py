import ctypes
import warnings
from collections import namedtuple
from pyglet.util import asbytes, asstr
from pyglet.font import base
from pyglet import image
from pyglet.font.fontconfig import get_fontconfig
from pyglet.font.freetype_lib import *
def _get_font_family_from_ttf(self):
    return
    if self.face_flags & FT_FACE_FLAG_SFNT:
        name = FT_SfntName()
        for i in range(FT_Get_Sfnt_Name_Count(self.ft_face)):
            try:
                FT_Get_Sfnt_Name(self.ft_face, i, name)
                if not (name.platform_id == TT_PLATFORM_MICROSOFT and name.encoding_id == TT_MS_ID_UNICODE_CS):
                    continue
                self._name = name.string.decode('utf-16be', 'ignore')
            except:
                continue