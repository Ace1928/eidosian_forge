from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
class FontConfig:

    def __init__(self):
        self._fontconfig = self._load_fontconfig_library()
        self._search_cache = OrderedDict()
        self._cache_size = 20

    def dispose(self):
        while len(self._search_cache) > 0:
            self._search_cache.popitem().dispose()
        self._fontconfig.FcFini()
        self._fontconfig = None

    def create_search_pattern(self):
        return FontConfigSearchPattern(self._fontconfig)

    def find_font(self, name, size=12, bold=False, italic=False):
        result = self._get_from_search_cache(name, size, bold, italic)
        if result:
            return result
        search_pattern = self.create_search_pattern()
        search_pattern.name = name
        search_pattern.size = size
        search_pattern.bold = bold
        search_pattern.italic = italic
        result = search_pattern.match()
        self._add_to_search_cache(search_pattern, result)
        search_pattern.dispose()
        return result

    def have_font(self, name):
        result = self.find_font(name)
        if result:
            if name and result.name and (result.name.lower() != name.lower()):
                return False
            return True
        else:
            return False

    def char_index(self, ft_face, character):
        return self._fontconfig.FcFreeTypeCharIndex(ft_face, ord(character))

    def _add_to_search_cache(self, search_pattern, result_pattern):
        self._search_cache[search_pattern.name, search_pattern.size, search_pattern.bold, search_pattern.italic] = result_pattern
        if len(self._search_cache) > self._cache_size:
            self._search_cache.popitem(last=False)[1].dispose()

    def _get_from_search_cache(self, name, size, bold, italic):
        result = self._search_cache.get((name, size, bold, italic), None)
        if result and result.is_valid:
            return result
        else:
            return None

    @staticmethod
    def _load_fontconfig_library():
        fontconfig = pyglet.lib.load_library('fontconfig')
        fontconfig.FcInit()
        fontconfig.FcPatternBuild.restype = c_void_p
        fontconfig.FcPatternCreate.restype = c_void_p
        fontconfig.FcFontMatch.restype = c_void_p
        fontconfig.FcFreeTypeCharIndex.restype = c_uint
        fontconfig.FcPatternAddDouble.argtypes = [c_void_p, c_char_p, c_double]
        fontconfig.FcPatternAddInteger.argtypes = [c_void_p, c_char_p, c_int]
        fontconfig.FcPatternAddString.argtypes = [c_void_p, c_char_p, c_char_p]
        fontconfig.FcConfigSubstitute.argtypes = [c_void_p, c_void_p, c_int]
        fontconfig.FcDefaultSubstitute.argtypes = [c_void_p]
        fontconfig.FcFontMatch.argtypes = [c_void_p, c_void_p, c_void_p]
        fontconfig.FcPatternDestroy.argtypes = [c_void_p]
        fontconfig.FcPatternGetFTFace.argtypes = [c_void_p, c_char_p, c_int, c_void_p]
        fontconfig.FcPatternGet.argtypes = [c_void_p, c_char_p, c_int, c_void_p]
        return fontconfig