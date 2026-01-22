from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class FzFont(object):
    """
    Wrapper class for struct `fz_font`.
    An abstract font handle.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_advance_glyph(self, glyph, wmode):
        """
        Class-aware wrapper for `::fz_advance_glyph()`.
        	Return the advance for a given glyph.

        	font: The font to look for the glyph in.

        	glyph: The glyph to find the advance for.

        	wmode: 1 for vertical mode, 0 for horizontal.

        	Returns the advance for the glyph.
        """
        return _mupdf.FzFont_fz_advance_glyph(self, glyph, wmode)

    def fz_bound_glyph(self, gid, trm):
        """
        Class-aware wrapper for `::fz_bound_glyph()`.
        	Return a bbox for a given glyph in a font.

        	font: The font to look for the glyph in.

        	gid: The glyph to bound.

        	trm: The matrix to apply to the glyph before bounding.

        	Returns rectangle by value containing the bounds of the given
        	glyph.
        """
        return _mupdf.FzFont_fz_bound_glyph(self, gid, trm)

    def fz_decouple_type3_font(self, t3doc):
        """ Class-aware wrapper for `::fz_decouple_type3_font()`."""
        return _mupdf.FzFont_fz_decouple_type3_font(self, t3doc)

    def fz_encode_character(self, unicode):
        """
        Class-aware wrapper for `::fz_encode_character()`.
        	Find the glyph id for a given unicode
        	character within a font.

        	font: The font to look for the unicode character in.

        	unicode: The unicode character to encode.

        	Returns the glyph id for the given unicode value, or 0 if
        	unknown.
        """
        return _mupdf.FzFont_fz_encode_character(self, unicode)

    def fz_encode_character_by_glyph_name(self, glyphname):
        """
        Class-aware wrapper for `::fz_encode_character_by_glyph_name()`.
        	Encode character.

        	Either by direct lookup of glyphname within a font, or, failing
        	that, by mapping glyphname to unicode and thence to the glyph
        	index within the given font.

        	Returns zero for type3 fonts.
        """
        return _mupdf.FzFont_fz_encode_character_by_glyph_name(self, glyphname)

    def fz_encode_character_sc(self, unicode):
        """
        Class-aware wrapper for `::fz_encode_character_sc()`.
        	Encode character, preferring small-caps variant if available.

        	font: The font to look for the unicode character in.

        	unicode: The unicode character to encode.

        	Returns the glyph id for the given unicode value, or 0 if
        	unknown.
        """
        return _mupdf.FzFont_fz_encode_character_sc(self, unicode)

    def fz_encode_character_with_fallback(self, unicode, script, language, out_font):
        """
        Class-aware wrapper for `::fz_encode_character_with_fallback()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_encode_character_with_fallback(int unicode, int script, int language, ::fz_font **out_font)` => `(int)`

        	Find the glyph id for
        	a given unicode character within a font, falling back to
        	an alternative if not found.

        	font: The font to look for the unicode character in.

        	unicode: The unicode character to encode.

        	script: The script in use.

        	language: The language in use.

        	out_font: The font handle in which the given glyph represents
        	the requested unicode character. The caller does not own the
        	reference it is passed, so should call fz_keep_font if it is
        	not simply to be used immediately.

        	Returns the glyph id for the given unicode value in the supplied
        	font (and sets *out_font to font) if it is present. Otherwise
        	an alternative fallback font (based on script/language) is
        	searched for. If the glyph is found therein, *out_font is set
        	to this reference, and the glyph reference is returned. If it
        	cannot be found anywhere, the function returns 0.
        """
        return _mupdf.FzFont_fz_encode_character_with_fallback(self, unicode, script, language, out_font)

    def fz_extract_ttf_from_ttc(self):
        """ Class-aware wrapper for `::fz_extract_ttf_from_ttc()`."""
        return _mupdf.FzFont_fz_extract_ttf_from_ttc(self)

    def fz_font_ascender(self):
        """
        Class-aware wrapper for `::fz_font_ascender()`.
        	Retrieve font ascender in ems.
        """
        return _mupdf.FzFont_fz_font_ascender(self)

    def fz_font_bbox(self):
        """
        Class-aware wrapper for `::fz_font_bbox()`.
        	Retrieve the font bbox.

        	font: The font to query.

        	Returns the font bbox by value; it is valid only if
        	fz_font_flags(font)->invalid_bbox is zero.
        """
        return _mupdf.FzFont_fz_font_bbox(self)

    def fz_font_descender(self):
        """
        Class-aware wrapper for `::fz_font_descender()`.
        	Retrieve font descender in ems.
        """
        return _mupdf.FzFont_fz_font_descender(self)

    def fz_font_digest(self, digest):
        """
        Class-aware wrapper for `::fz_font_digest()`.
        	Retrieve the MD5 digest for the font's data.
        """
        return _mupdf.FzFont_fz_font_digest(self, digest)

    def fz_font_ft_face(self):
        """
        Class-aware wrapper for `::fz_font_ft_face()`.
        	Retrieve the FT_Face handle
        	for the font.

        	font: The font to query

        	Returns the FT_Face handle for the font, or NULL
        	if not a freetype handled font. (Cast to void *
        	to avoid nasty header exposure).
        """
        return _mupdf.FzFont_fz_font_ft_face(self)

    def fz_font_is_bold(self):
        """
        Class-aware wrapper for `::fz_font_is_bold()`.
        	Query whether the font flags say that this font is bold.
        """
        return _mupdf.FzFont_fz_font_is_bold(self)

    def fz_font_is_italic(self):
        """
        Class-aware wrapper for `::fz_font_is_italic()`.
        	Query whether the font flags say that this font is italic.
        """
        return _mupdf.FzFont_fz_font_is_italic(self)

    def fz_font_is_monospaced(self):
        """
        Class-aware wrapper for `::fz_font_is_monospaced()`.
        	Query whether the font flags say that this font is monospaced.
        """
        return _mupdf.FzFont_fz_font_is_monospaced(self)

    def fz_font_is_serif(self):
        """
        Class-aware wrapper for `::fz_font_is_serif()`.
        	Query whether the font flags say that this font is serif.
        """
        return _mupdf.FzFont_fz_font_is_serif(self)

    def fz_font_name(self):
        """
        Class-aware wrapper for `::fz_font_name()`.
        	Retrieve a pointer to the name of the font.

        	font: The font to query.

        	Returns a pointer to an internal copy of the font name.
        	Will never be NULL, but may be the empty string.
        """
        return _mupdf.FzFont_fz_font_name(self)

    def fz_font_t3_procs(self):
        """
        Class-aware wrapper for `::fz_font_t3_procs()`.
        	Retrieve the Type3 procs
        	for a font.

        	font: The font to query

        	Returns the t3_procs pointer. Will be NULL for a
        	non type-3 font.
        """
        return _mupdf.FzFont_fz_font_t3_procs(self)

    def fz_get_glyph_name(self, glyph, buf, size):
        """
        Class-aware wrapper for `::fz_get_glyph_name()`.
        	Find the name of a glyph

        	font: The font to look for the glyph in.

        	glyph: The glyph id to look for.

        	buf: Pointer to a buffer for the name to be inserted into.

        	size: The size of the buffer.

        	If a font contains a name table, then the name of the glyph
        	will be returned in the supplied buffer. Otherwise a name
        	is synthesised. The name will be truncated to fit in
        	the buffer.
        """
        return _mupdf.FzFont_fz_get_glyph_name(self, glyph, buf, size)

    def fz_get_glyph_name2(self, glyph):
        """
        Class-aware wrapper for `::fz_get_glyph_name2()`.
        C++ alternative to fz_get_glyph_name() that returns information in a std::string.
        """
        return _mupdf.FzFont_fz_get_glyph_name2(self, glyph)

    def fz_glyph_cacheable(self, gid):
        """
        Class-aware wrapper for `::fz_glyph_cacheable()`.
        	Determine if a given glyph in a font
        	is cacheable. Certain glyphs in a type 3 font cannot safely
        	be cached, as their appearance depends on the enclosing
        	graphic state.

        	font: The font to look for the glyph in.

        	gif: The glyph to query.

        	Returns non-zero if cacheable, 0 if not.
        """
        return _mupdf.FzFont_fz_glyph_cacheable(self, gid)

    def fz_measure_string(self, trm, s, wmode, bidi_level, markup_dir, language):
        """
        Class-aware wrapper for `::fz_measure_string()`.
        	Measure the advance width of a UTF8 string should it be added to a text object.

        	This uses the same layout algorithms as fz_show_string, and can be used
        	to calculate text alignment adjustments.
        """
        return _mupdf.FzFont_fz_measure_string(self, trm, s, wmode, bidi_level, markup_dir, language)

    def fz_outline_glyph(self, gid, ctm):
        """
        Class-aware wrapper for `::fz_outline_glyph()`.
        	Look a glyph up from a font, and return the outline of the
        	glyph using the given transform.

        	The caller owns the returned path, and so is responsible for
        	ensuring that it eventually gets dropped.
        """
        return _mupdf.FzFont_fz_outline_glyph(self, gid, ctm)

    def fz_prepare_t3_glyph(self, gid):
        """
        Class-aware wrapper for `::fz_prepare_t3_glyph()`.
        	Force a type3 font to cache the displaylist for a given glyph
        	id.

        	This caching can involve reading the underlying file, so must
        	happen ahead of time, so we aren't suddenly forced to read the
        	file while playing a displaylist back.
        """
        return _mupdf.FzFont_fz_prepare_t3_glyph(self, gid)

    def fz_render_glyph_pixmap(self, gid, ctm, scissor, aa):
        """
        Class-aware wrapper for `::fz_render_glyph_pixmap()`.
        	Create a pixmap containing a rendered glyph.

        	Lookup gid from font, clip it with scissor, and rendering it
        	with aa bits of antialiasing into a new pixmap.

        	The caller takes ownership of the pixmap and so must free it.

        	Note: This function is no longer used for normal rendering
        	operations, and is kept around just because we use it in the
        	app. It should be considered "at risk" of removal from the API.
        """
        return _mupdf.FzFont_fz_render_glyph_pixmap(self, gid, ctm, scissor, aa)

    def fz_run_t3_glyph(self, gid, trm, dev):
        """
        Class-aware wrapper for `::fz_run_t3_glyph()`.
        	Run a glyph from a Type3 font to
        	a given device.

        	font: The font to find the glyph in.

        	gid: The glyph to run.

        	trm: The transform to apply.

        	dev: The device to render onto.
        """
        return _mupdf.FzFont_fz_run_t3_glyph(self, gid, trm, dev)

    def fz_set_font_bbox(self, xmin, ymin, xmax, ymax):
        """
        Class-aware wrapper for `::fz_set_font_bbox()`.
        	Set the font bbox.

        	font: The font to set the bbox for.

        	xmin, ymin, xmax, ymax: The bounding box.
        """
        return _mupdf.FzFont_fz_set_font_bbox(self, xmin, ymin, xmax, ymax)

    def fz_set_font_embedding(self, embed):
        """
        Class-aware wrapper for `::fz_set_font_embedding()`.
        	Control whether a given font should be embedded or not when writing.
        """
        return _mupdf.FzFont_fz_set_font_embedding(self, embed)

    def pdf_font_writing_supported(self):
        """ Class-aware wrapper for `::pdf_font_writing_supported()`."""
        return _mupdf.FzFont_pdf_font_writing_supported(self)

    def pdf_layout_fit_text(self, lang, str, bounds):
        """ Class-aware wrapper for `::pdf_layout_fit_text()`."""
        return _mupdf.FzFont_pdf_layout_fit_text(self, lang, str, bounds)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_base14_font()`.
        		Create a new font from one of the built-in fonts.


        |

        *Overload 2:*
         Constructor using `fz_new_builtin_font()`.

        |

        *Overload 3:*
         Constructor using `fz_new_cjk_font()`.

        |

        *Overload 4:*
         Constructor using `fz_new_font_from_buffer()`.
        		Create a new font from a font file in a fz_buffer.

        		Fonts created in this way, will be eligible for embedding by default.

        		name: Name of font (leave NULL to use name from font).

        		buffer: Buffer to load from.

        		index: Which font from the file to load (0 for default).

        		use_glyph_box: 1 if we should use the glyph bbox, 0 otherwise.

        		Returns new font handle, or throws exception on error.


        |

        *Overload 5:*
         Constructor using `fz_new_font_from_file()`.
        		Create a new font from a font file.

        		Fonts created in this way, will be eligible for embedding by default.

        		name: Name of font (leave NULL to use name from font).

        		path: File path to load from.

        		index: Which font from the file to load (0 for default).

        		use_glyph_box: 1 if we should use the glyph bbox, 0 otherwise.

        		Returns new font handle, or throws exception on error.


        |

        *Overload 6:*
         Constructor using `fz_new_font_from_memory()`.
        		Create a new font from a font file in memory.

        		Fonts created in this way, will be eligible for embedding by default.

        		name: Name of font (leave NULL to use name from font).

        		data: Pointer to the font file data.

        		len: Length of the font file data.

        		index: Which font from the file to load (0 for default).

        		use_glyph_box: 1 if we should use the glyph bbox, 0 otherwise.

        		Returns new font handle, or throws exception on error.


        |

        *Overload 7:*
         Constructor using `fz_new_type3_font()`.
        		Create a new (empty) type3 font.

        		name: Name of font (or NULL).

        		matrix: Font matrix.

        		Returns a new font handle, or throws exception on
        		allocation failure.


        |

        *Overload 8:*
         Copy constructor using `fz_keep_font()`.

        |

        *Overload 9:*
         Default constructor, sets `m_internal` to null.

        |

        *Overload 10:*
         Constructor using raw copy of pre-existing `::fz_font`.
        """
        _mupdf.FzFont_swiginit(self, _mupdf.new_FzFont(*args))
    __swig_destroy__ = _mupdf.delete_FzFont

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzFont_m_internal_value(self)
    m_internal = property(_mupdf.FzFont_m_internal_get, _mupdf.FzFont_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzFont_s_num_instances_get, _mupdf.FzFont_s_num_instances_set)