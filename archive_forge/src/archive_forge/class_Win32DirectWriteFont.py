import copy
import os
import pathlib
import platform
from ctypes import *
from typing import List, Optional, Tuple
import math
import pyglet
from pyglet.font import base
from pyglet.image.codecs.wic import IWICBitmap, WICDecoder, GUID_WICPixelFormat32bppPBGRA
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
class Win32DirectWriteFont(base.Font):
    _custom_collection = None
    _write_factory = None
    _font_loader = None
    _font_builder = None
    _font_set = None
    _font_collection_loader = None
    _font_cache = []
    _font_loader_key = None
    _default_name = 'Segoe UI'
    _glyph_renderer = None
    _empty_glyph = None
    _zero_glyph = None
    glyph_renderer_class = DirectWriteGlyphRenderer
    texture_internalformat = pyglet.gl.GL_RGBA

    def __init__(self, name, size, bold=False, italic=False, stretch=False, dpi=None, locale=None):
        self._filename: Optional[str] = None
        self._advance_cache = {}
        super(Win32DirectWriteFont, self).__init__()
        if not name:
            name = self._default_name
        self._name = name
        self.bold = bold
        self.size = size
        self.italic = italic
        self.stretch = stretch
        self.dpi = dpi
        self.locale = locale
        if self.locale is None:
            self.locale = ''
            self.rtl = False
        if self.dpi is None:
            self.dpi = 96
        self._real_size = self.size * self.dpi // 72
        if self.bold:
            if type(self.bold) is str:
                self._weight = name_to_weight[self.bold]
            else:
                self._weight = DWRITE_FONT_WEIGHT_BOLD
        else:
            self._weight = DWRITE_FONT_WEIGHT_NORMAL
        if self.italic:
            if type(self.italic) is str:
                self._style = name_to_style[self.italic]
            else:
                self._style = DWRITE_FONT_STYLE_ITALIC
        else:
            self._style = DWRITE_FONT_STYLE_NORMAL
        if self.stretch:
            if type(self.stretch) is str:
                self._stretch = name_to_stretch[self.stretch]
            else:
                self._stretch = DWRITE_FONT_STRETCH_EXPANDED
        else:
            self._stretch = DWRITE_FONT_STRETCH_NORMAL
        self._font_index, self._collection = self.get_collection(name)
        write_font = None
        if pyglet.options['dw_legacy_naming']:
            if self._font_index is None and self._collection is None:
                write_font, self._collection = self.find_font_face(name, self._weight, self._style, self._stretch)
        assert self._collection is not None, f"Font: '{name}' not found in loaded or system font collection."
        if self._font_index is not None:
            font_family = IDWriteFontFamily1()
            self._collection.GetFontFamily(self._font_index, byref(font_family))
            write_font = IDWriteFont()
            font_family.GetFirstMatchingFont(self._weight, self._stretch, self._style, byref(write_font))
        self._text_format = IDWriteTextFormat()
        self._write_factory.CreateTextFormat(self._name, self._collection, self._weight, self._style, self._stretch, self._real_size, create_unicode_buffer(self.locale), byref(self._text_format))
        font_face = IDWriteFontFace()
        write_font.CreateFontFace(byref(font_face))
        self.font_face = IDWriteFontFace1()
        font_face.QueryInterface(IID_IDWriteFontFace1, byref(self.font_face))
        self._font_metrics = DWRITE_FONT_METRICS()
        self.font_face.GetMetrics(byref(self._font_metrics))
        self.font_scale_ratio = self._real_size / self._font_metrics.designUnitsPerEm
        self.ascent = math.ceil(self._font_metrics.ascent * self.font_scale_ratio)
        self.descent = -round(self._font_metrics.descent * self.font_scale_ratio)
        self.max_glyph_height = (self._font_metrics.ascent + self._font_metrics.descent) * self.font_scale_ratio
        self.line_gap = self._font_metrics.lineGap * self.font_scale_ratio
        self._fallback = None
        if WINDOWS_8_1_OR_GREATER:
            self._fallback = IDWriteFontFallback()
            self._write_factory.GetSystemFontFallback(byref(self._fallback))
        else:
            assert _debug_print('Windows 8.1+ is required for font fallback. Colored glyphs cannot be omitted.')

    @property
    def filename(self):
        """Returns a filename associated with the font face.
        Note: Capable of returning more than 1 file in the future, but will do just one for now."""
        if self._filename is not None:
            return self._filename
        file_ct = UINT32()
        self.font_face.GetFiles(byref(file_ct), None)
        font_files = (IDWriteFontFile * file_ct.value)()
        self.font_face.GetFiles(byref(file_ct), font_files)
        self._filename = 'Not Available'
        pff = font_files[0]
        key_data = c_void_p()
        ff_key_size = UINT32()
        pff.GetReferenceKey(byref(key_data), byref(ff_key_size))
        loader = IDWriteFontFileLoader()
        pff.GetLoader(byref(loader))
        try:
            local_loader = IDWriteLocalFontFileLoader()
            loader.QueryInterface(IID_IDWriteLocalFontFileLoader, byref(local_loader))
        except OSError:
            loader.Release()
            pff.Release()
            return self._filename
        path_len = UINT32()
        local_loader.GetFilePathLengthFromKey(key_data, ff_key_size, byref(path_len))
        buffer = create_unicode_buffer(path_len.value + 1)
        local_loader.GetFilePathFromKey(key_data, ff_key_size, buffer, len(buffer))
        loader.Release()
        local_loader.Release()
        pff.Release()
        self._filename = pathlib.PureWindowsPath(buffer.value).as_posix()
        return self._filename

    @property
    def name(self):
        return self._name

    def render_to_image(self, text, width=10000, height=80):
        """This process takes Pyglet out of the equation and uses only DirectWrite to shape and render text.
        This may allow more accurate fonts (bidi, rtl, etc) in very special circumstances at the cost of
        additional texture space.

        :Parameters:
            `text` : str
                String of text to render.

        :rtype: `ImageData`
        :return: An image of the text.
        """
        if not self._glyph_renderer:
            self._glyph_renderer = self.glyph_renderer_class(self)
        return self._glyph_renderer.render_to_image(text, width, height)

    def copy_glyph(self, glyph, advance, offset):
        """This takes the existing glyph texture and puts it into a new Glyph with a new advance.
        Texture memory is shared between both glyphs."""
        new_glyph = base.Glyph(glyph.x, glyph.y, glyph.z, glyph.width, glyph.height, glyph.owner)
        new_glyph.set_bearings(glyph.baseline, glyph.lsb, advance, offset.advanceOffset, offset.ascenderOffset)
        return new_glyph

    def _render_layout_glyph(self, text_buffer, i, clusters, check_color=True):
        text_length = clusters.count(i)
        text_index = clusters.index(i)
        actual_text = text_buffer[text_index:text_index + text_length]
        if actual_text not in self.glyphs:
            glyph = self._glyph_renderer.render_using_layout(text_buffer[text_index:text_index + text_length])
            if glyph:
                if check_color and self._glyph_renderer.draw_options & D2D1_DRAW_TEXT_OPTIONS_ENABLE_COLOR_FONT:
                    fb_ff = self._get_fallback_font_face(text_index, text_length)
                    if fb_ff:
                        glyph.colored = self.is_fallback_str_colored(fb_ff, actual_text)
            else:
                glyph = self._empty_glyph
            self.glyphs[actual_text] = glyph
        return self.glyphs[actual_text]

    def is_fallback_str_colored(self, font_face, text):
        indice = UINT16()
        code_points = (UINT32 * len(text))(*[ord(c) for c in text])
        font_face.GetGlyphIndices(code_points, len(text), byref(indice))
        new_indice = (UINT16 * 1)(indice)
        new_advance = (FLOAT * 1)(100)
        offset = (DWRITE_GLYPH_OFFSET * 1)()
        run = self._glyph_renderer._get_single_glyph_run(font_face, self._real_size, new_indice, new_advance, offset, False, False)
        return self._glyph_renderer.is_color_run(run)

    def _get_fallback_font_face(self, text_index, text_length):
        if WINDOWS_8_1_OR_GREATER:
            out_length = UINT32()
            fb_font = IDWriteFont()
            scale = FLOAT()
            self._fallback.MapCharacters(self._glyph_renderer._text_analysis, text_index, text_length, None, None, self._weight, self._style, self._stretch, byref(out_length), byref(fb_font), byref(scale))
            if fb_font:
                fb_font_face = IDWriteFontFace()
                fb_font.CreateFontFace(byref(fb_font_face))
                return fb_font_face
        return None

    def get_glyphs_no_shape(self, text):
        """This differs in that it does not attempt to shape the text at all. May be useful in cases where your font
        has no special shaping requirements, spacing is the same, or some other reason where faster performance is
        wanted and you can get away with this."""
        if not self._glyph_renderer:
            self._glyph_renderer = self.glyph_renderer_class(self)
            self._empty_glyph = self._glyph_renderer.render_using_layout(' ')
        glyphs = []
        for c in text:
            if c == '\t':
                c = ' '
            if c not in self.glyphs:
                self.glyphs[c] = self._glyph_renderer.render_using_layout(c)
                if not self.glyphs[c]:
                    self.glyphs[c] = self._empty_glyph
            glyphs.append(self.glyphs[c])
        return glyphs

    def get_glyphs(self, text):
        if not self._glyph_renderer:
            self._glyph_renderer = self.glyph_renderer_class(self)
            self._empty_glyph = self._glyph_renderer.render_using_layout(' ')
            self._zero_glyph = self._glyph_renderer.create_zero_glyph()
        text_buffer, actual_count, indices, advances, offsets, clusters = self._glyph_renderer.get_string_info(text, self.font_face)
        metrics = self._glyph_renderer.get_glyph_metrics(self.font_face, indices, actual_count)
        formatted_clusters = list(clusters)
        for i in range(actual_count):
            advances[i] *= self.font_scale_ratio
        for i in range(actual_count):
            offsets[i].advanceOffset *= self.font_scale_ratio
            offsets[i].ascenderOffset *= self.font_scale_ratio
        glyphs = []
        substitutions = {}
        for idx in clusters:
            ct = formatted_clusters.count(idx)
            if ct > 1:
                substitutions[idx] = ct - 1
        for i in range(actual_count):
            indice = indices[i]
            if indice == 0:
                glyph = self._render_layout_glyph(text_buffer, i, formatted_clusters)
                glyphs.append(glyph)
            else:
                advance_key = (indice, advances[i], offsets[i].advanceOffset, offsets[i].ascenderOffset)
                if indice in self.glyphs:
                    if advance_key in self._advance_cache:
                        glyph = self._advance_cache[advance_key]
                    else:
                        glyph = self.copy_glyph(self.glyphs[indice], advances[i], offsets[i])
                        self._advance_cache[advance_key] = glyph
                else:
                    glyph = self._glyph_renderer.render_single_glyph(self.font_face, indice, advances[i], offsets[i], metrics[i])
                    if glyph is None:
                        glyph = self._render_layout_glyph(text_buffer, i, formatted_clusters, check_color=False)
                        glyph.colored = True
                    self.glyphs[indice] = glyph
                    self._advance_cache[advance_key] = glyph
                glyphs.append(glyph)
            if i in substitutions:
                for _ in range(substitutions[i]):
                    glyphs.append(self._zero_glyph)
        return glyphs

    def create_text_layout(self, text):
        text_buffer = create_unicode_buffer(text)
        text_layout = IDWriteTextLayout()
        hr = self._write_factory.CreateTextLayout(text_buffer, len(text_buffer), self._text_format, 10000, 80, byref(text_layout))
        return text_layout

    @classmethod
    def _initialize_direct_write(cls):
        """ All direct write fonts needs factory access as well as the loaders."""
        if WINDOWS_10_CREATORS_UPDATE_OR_GREATER:
            cls._write_factory = IDWriteFactory5()
            guid = IID_IDWriteFactory5
        elif WINDOWS_8_1_OR_GREATER:
            cls._write_factory = IDWriteFactory2()
            guid = IID_IDWriteFactory2
        else:
            cls._write_factory = IDWriteFactory()
            guid = IID_IDWriteFactory
        DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, guid, byref(cls._write_factory))

    @classmethod
    def _initialize_custom_loaders(cls):
        """Initialize the loaders needed to load custom fonts."""
        if WINDOWS_10_CREATORS_UPDATE_OR_GREATER:
            cls._font_loader = IDWriteInMemoryFontFileLoader()
            cls._write_factory.CreateInMemoryFontFileLoader(byref(cls._font_loader))
            cls._write_factory.RegisterFontFileLoader(cls._font_loader)
            cls._font_builder = IDWriteFontSetBuilder1()
            cls._write_factory.CreateFontSetBuilder1(byref(cls._font_builder))
        else:
            cls._font_loader = LegacyFontFileLoader()
            cls._write_factory.RegisterFontFileLoader(cls._font_loader.as_interface(IDWriteFontFileLoader_LI))
            cls._font_collection_loader = LegacyCollectionLoader(cls._write_factory, cls._font_loader)
            cls._write_factory.RegisterFontCollectionLoader(cls._font_collection_loader)
            cls._font_loader_key = cast(create_unicode_buffer('legacy_font_loader'), c_void_p)

    @classmethod
    def add_font_data(cls, data):
        if not cls._write_factory:
            cls._initialize_direct_write()
        if not cls._font_loader:
            cls._initialize_custom_loaders()
        if WINDOWS_10_CREATORS_UPDATE_OR_GREATER:
            font_file = IDWriteFontFile()
            hr = cls._font_loader.CreateInMemoryFontFileReference(cls._write_factory, data, len(data), None, byref(font_file))
            hr = cls._font_builder.AddFontFile(font_file)
            if hr != 0:
                raise Exception('This font file data is not not a font or unsupported.')
            if cls._custom_collection:
                cls._font_set.Release()
                cls._custom_collection.Release()
            cls._font_set = IDWriteFontSet()
            cls._font_builder.CreateFontSet(byref(cls._font_set))
            cls._custom_collection = IDWriteFontCollection1()
            cls._write_factory.CreateFontCollectionFromFontSet(cls._font_set, byref(cls._custom_collection))
        else:
            cls._font_cache.append(data)
            if cls._custom_collection:
                cls._custom_collection = None
                cls._write_factory.UnregisterFontCollectionLoader(cls._font_collection_loader)
                cls._write_factory.UnregisterFontFileLoader(cls._font_loader)
                cls._font_loader = LegacyFontFileLoader()
                cls._font_collection_loader = LegacyCollectionLoader(cls._write_factory, cls._font_loader)
                cls._write_factory.RegisterFontCollectionLoader(cls._font_collection_loader)
                cls._write_factory.RegisterFontFileLoader(cls._font_loader.as_interface(IDWriteFontFileLoader_LI))
            cls._font_collection_loader.AddFontData(cls._font_cache)
            cls._custom_collection = IDWriteFontCollection()
            cls._write_factory.CreateCustomFontCollection(cls._font_collection_loader, cls._font_loader_key, sizeof(cls._font_loader_key), byref(cls._custom_collection))

    @classmethod
    def get_collection(cls, font_name) -> Tuple[Optional[int], Optional[IDWriteFontCollection1]]:
        """Returns which collection this font belongs to (system or custom collection), as well as its index in the
        collection."""
        if not cls._write_factory:
            cls._initialize_direct_write()
        'Returns a collection the font_name belongs to.'
        font_index = UINT()
        font_exists = BOOL()
        if cls._custom_collection:
            cls._custom_collection.FindFamilyName(create_unicode_buffer(font_name), byref(font_index), byref(font_exists))
            if font_exists.value:
                return (font_index.value, cls._custom_collection)
        sys_collection = IDWriteFontCollection()
        if not font_exists.value:
            cls._write_factory.GetSystemFontCollection(byref(sys_collection), 1)
            sys_collection.FindFamilyName(create_unicode_buffer(font_name), byref(font_index), byref(font_exists))
            if font_exists.value:
                return (font_index.value, sys_collection)
        return (None, None)

    @classmethod
    def find_font_face(cls, font_name, bold, italic, stretch) -> Tuple[Optional[IDWriteFont], Optional[IDWriteFontCollection]]:
        """This will search font collections for legacy RBIZ names. However, matching to bold, italic, stretch is
        problematic in that there are many values. We parse the font name looking for matches to the name database,
        and attempt to pick the closest match.
        This will search all fonts on the system and custom loaded, and all of their font faces. Returns a collection
        and IDWriteFont if successful.
        """
        p_bold, p_italic, p_stretch = cls.parse_name(font_name, bold, italic, stretch)
        _debug_print(f"directwrite: '{font_name}' not found. Attempting legacy name lookup in all collections.")
        collection_idx = cls.find_legacy_font(cls._custom_collection, font_name, p_bold, p_italic, p_stretch)
        if collection_idx is not None:
            return (collection_idx, cls._custom_collection)
        sys_collection = IDWriteFontCollection()
        cls._write_factory.GetSystemFontCollection(byref(sys_collection), 1)
        collection_idx = cls.find_legacy_font(sys_collection, font_name, p_bold, p_italic, p_stretch)
        if collection_idx is not None:
            return (collection_idx, sys_collection)
        return (None, None)

    @classmethod
    def have_font(cls, name: str):
        if cls.get_collection(name)[0] is not None:
            return True
        return False

    @staticmethod
    def parse_name(font_name: str, weight: int, style: int, stretch: int):
        """Attempt at parsing any special names in a font for legacy checks. Takes the first found."""
        font_name = font_name.lower()
        split_name = font_name.split(' ')
        found_weight = weight
        found_style = style
        found_stretch = stretch
        if len(split_name) > 1:
            for name, value in name_to_weight.items():
                if name in split_name:
                    found_weight = value
                    break
            for name, value in name_to_style.items():
                if name in split_name:
                    found_style = value
                    break
            for name, value in name_to_stretch.items():
                if name in split_name:
                    found_stretch = value
                    break
        return (found_weight, found_style, found_stretch)

    @staticmethod
    def find_legacy_font(collection: IDWriteFontCollection, font_name: str, bold, italic, stretch, full_debug=False) -> Optional[IDWriteFont]:
        coll_count = collection.GetFontFamilyCount()
        assert _debug_print(f'directwrite: Found {coll_count} fonts in collection.')
        locale = get_system_locale()
        for i in range(coll_count):
            family = IDWriteFontFamily()
            collection.GetFontFamily(i, byref(family))
            family_name_str = IDWriteLocalizedStrings()
            family.GetFamilyNames(byref(family_name_str))
            family_names = Win32DirectWriteFont.unpack_localized_string(family_name_str, locale)
            family_name = family_names[0]
            if family_name[0] != font_name[0]:
                family.Release()
                continue
            assert _debug_print(f'directwrite: Inspecting family name: {family_name}')
            ft_ct = family.GetFontCount()
            face_names = []
            matches = []
            for j in range(ft_ct):
                temp_ft = IDWriteFont()
                family.GetFont(j, byref(temp_ft))
                if _debug_font and full_debug:
                    fc_str = IDWriteLocalizedStrings()
                    temp_ft.GetFaceNames(byref(fc_str))
                    strings = Win32DirectWriteFont.unpack_localized_string(fc_str, locale)
                    face_names.extend(strings)
                    print(f'directwrite: Face names found: {strings}')
                compat_names = IDWriteLocalizedStrings()
                exists = BOOL()
                temp_ft.GetInformationalStrings(DWRITE_INFORMATIONAL_STRING_WIN32_FAMILY_NAMES, byref(compat_names), byref(exists))
                match_found = False
                if exists.value != 0:
                    for compat_name in Win32DirectWriteFont.unpack_localized_string(compat_names, locale):
                        if compat_name == font_name:
                            assert _debug_print(f"Found legacy name '{font_name}' as '{family_name}' in font face '{j}' (collection id #{i}).")
                            match_found = True
                            matches.append((temp_ft.GetWeight(), temp_ft.GetStyle(), temp_ft.GetStretch(), temp_ft))
                            break
                if not match_found:
                    temp_ft.Release()
            family.Release()
            if matches:
                write_font = Win32DirectWriteFont.match_closest_font(matches, bold, italic, stretch)
                for match in matches:
                    if match[3] != write_font:
                        match[3].Release()
                return write_font
        return None

    @staticmethod
    def match_closest_font(font_list: List[Tuple[int, int, int, IDWriteFont]], bold: int, italic: int, stretch: int) -> Optional[IDWriteFont]:
        """Match the closest font to the parameters specified. If a full match is not found, a secondary match will be
        found based on similar features. This can probably be improved, but it is possible you could get a different
        font style than expected."""
        closest = []
        for match in font_list:
            f_weight, f_style, f_stretch, writefont = match
            if f_weight == bold and f_style == italic and (f_stretch == stretch):
                _debug_print(f'directwrite: full match found. (bold: {f_weight}, italic: {f_style}, stretch: {f_stretch})')
                return writefont
            prop_match = 0
            similar_match = 0
            if f_weight == bold:
                prop_match += 1
            elif bold != DWRITE_FONT_WEIGHT_NORMAL and f_weight != DWRITE_FONT_WEIGHT_NORMAL:
                similar_match += 1
            if f_style == italic:
                prop_match += 1
            elif italic != DWRITE_FONT_STYLE_NORMAL and f_style != DWRITE_FONT_STYLE_NORMAL:
                similar_match += 1
            if stretch == f_stretch:
                prop_match += 1
            elif stretch != DWRITE_FONT_STRETCH_NORMAL and f_stretch != DWRITE_FONT_STRETCH_NORMAL:
                similar_match += 1
            closest.append((prop_match, similar_match, *match))
        closest.sort(key=lambda fts: (fts[0], fts[1]), reverse=True)
        if closest:
            closest_match = closest[0]
            _debug_print(f'directwrite: falling back to partial match. (bold: {closest_match[2]}, italic: {closest_match[3]}, stretch: {closest_match[4]})')
            return closest_match[5]
        return None

    @staticmethod
    def unpack_localized_string(local_string: IDWriteLocalizedStrings, locale: str) -> List[str]:
        """Takes IDWriteLocalizedStrings and unpacks the strings inside of it into a list."""
        str_array_len = local_string.GetCount()
        strings = []
        for _ in range(str_array_len):
            string_size = UINT32()
            idx = Win32DirectWriteFont.get_localized_index(local_string, locale)
            local_string.GetStringLength(idx, byref(string_size))
            buffer_size = string_size.value
            buffer = create_unicode_buffer(buffer_size + 1)
            local_string.GetString(idx, buffer, len(buffer))
            strings.append(buffer.value)
        local_string.Release()
        return strings

    @staticmethod
    def get_localized_index(strings: IDWriteLocalizedStrings, locale: str):
        idx = UINT32()
        exists = BOOL()
        if locale:
            strings.FindLocaleName(locale, byref(idx), byref(exists))
            if not exists.value:
                strings.FindLocaleName('en-us', byref(idx), byref(exists))
                if not exists:
                    return 0
            return idx.value
        return 0