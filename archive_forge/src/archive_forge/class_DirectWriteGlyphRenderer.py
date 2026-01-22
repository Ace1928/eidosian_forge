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
class DirectWriteGlyphRenderer(base.GlyphRenderer):
    antialias_mode = D2D1_TEXT_ANTIALIAS_MODE_DEFAULT
    draw_options = D2D1_DRAW_TEXT_OPTIONS_ENABLE_COLOR_FONT if WINDOWS_8_1_OR_GREATER else D2D1_DRAW_TEXT_OPTIONS_NONE
    measuring_mode = DWRITE_MEASURING_MODE_NATURAL

    def __init__(self, font):
        self._render_target = None
        self._bitmap = None
        self._brush = None
        self._bitmap_dimensions = (0, 0)
        super(DirectWriteGlyphRenderer, self).__init__(font)
        self.font = font
        self._analyzer = IDWriteTextAnalyzer()
        self.font._write_factory.CreateTextAnalyzer(byref(self._analyzer))
        self._text_analysis = TextAnalysis()

    def render_to_image(self, text, width, height):
        """This process takes Pyglet out of the equation and uses only DirectWrite to shape and render text.
        This may allows more accurate fonts (bidi, rtl, etc) in very special circumstances."""
        text_buffer = create_unicode_buffer(text)
        text_layout = IDWriteTextLayout()
        self.font._write_factory.CreateTextLayout(text_buffer, len(text_buffer), self.font._text_format, width, height, byref(text_layout))
        layout_metrics = DWRITE_TEXT_METRICS()
        text_layout.GetMetrics(byref(layout_metrics))
        width, height = (int(math.ceil(layout_metrics.width)), int(math.ceil(layout_metrics.height)))
        bitmap = IWICBitmap()
        wic_decoder._factory.CreateBitmap(width, height, GUID_WICPixelFormat32bppPBGRA, WICBitmapCacheOnDemand, byref(bitmap))
        rt = ID2D1RenderTarget()
        d2d_factory.CreateWicBitmapRenderTarget(bitmap, default_target_properties, byref(rt))
        rt.SetTextAntialiasMode(self.antialias_mode)
        if not self._brush:
            self._brush = ID2D1SolidColorBrush()
        rt.CreateSolidColorBrush(white, None, byref(self._brush))
        rt.BeginDraw()
        rt.Clear(transparent)
        rt.DrawTextLayout(no_offset, text_layout, self._brush, self.draw_options)
        rt.EndDraw(None, None)
        rt.Release()
        image_data = wic_decoder.get_image(bitmap)
        return image_data

    def get_string_info(self, text, font_face):
        """Converts a string of text into a list of indices and advances used for shaping."""
        text_length = len(text.encode('utf-16-le')) // 2
        text_buffer = create_unicode_buffer(text, text_length)
        self._text_analysis.GenerateResults(self._analyzer, text_buffer, len(text_buffer))
        max_glyph_size = int(3 * text_length / 2 + 16)
        length = text_length
        clusters = (UINT16 * length)()
        text_props = (DWRITE_SHAPING_TEXT_PROPERTIES * length)()
        indices = (UINT16 * max_glyph_size)()
        glyph_props = (DWRITE_SHAPING_GLYPH_PROPERTIES * max_glyph_size)()
        actual_count = UINT32()
        self._analyzer.GetGlyphs(text_buffer, length, font_face, False, False, self._text_analysis._script, None, None, None, None, 0, max_glyph_size, clusters, text_props, indices, glyph_props, byref(actual_count))
        advances = (FLOAT * length)()
        offsets = (DWRITE_GLYPH_OFFSET * length)()
        self._analyzer.GetGlyphPlacements(text_buffer, clusters, text_props, text_length, indices, glyph_props, actual_count, font_face, self.font._font_metrics.designUnitsPerEm, False, False, self._text_analysis._script, self.font.locale, None, None, 0, advances, offsets)
        return (text_buffer, actual_count.value, indices, advances, offsets, clusters)

    def get_glyph_metrics(self, font_face, indices, count):
        """Returns a list of tuples with the following metrics per indice:
            (glyph width, glyph height, lsb, advanceWidth)
        """
        glyph_metrics = (DWRITE_GLYPH_METRICS * count)()
        font_face.GetDesignGlyphMetrics(indices, count, glyph_metrics, False)
        metrics_out = []
        for metric in glyph_metrics:
            glyph_width = metric.advanceWidth - metric.leftSideBearing - metric.rightSideBearing
            if glyph_width == 0:
                glyph_width = 1
            glyph_height = metric.advanceHeight - metric.topSideBearing - metric.bottomSideBearing
            lsb = metric.leftSideBearing
            bsb = metric.bottomSideBearing
            advance_width = metric.advanceWidth
            metrics_out.append((glyph_width, glyph_height, lsb, advance_width, bsb))
        return metrics_out

    def _get_single_glyph_run(self, font_face, size, indices, advances, offsets, sideways, bidi):
        run = DWRITE_GLYPH_RUN(font_face, size, 1, indices, advances, offsets, sideways, bidi)
        return run

    def is_color_run(self, run):
        """Will return True if the run contains a colored glyph."""
        try:
            if WINDOWS_10_CREATORS_UPDATE_OR_GREATER:
                enumerator = IDWriteColorGlyphRunEnumerator1()
                color = self.font._write_factory.TranslateColorGlyphRun4(no_offset, run, None, DWRITE_GLYPH_IMAGE_FORMATS_ALL, self.measuring_mode, None, 0, byref(enumerator))
            elif WINDOWS_8_1_OR_GREATER:
                enumerator = IDWriteColorGlyphRunEnumerator()
                color = self.font._write_factory.TranslateColorGlyphRun(0.0, 0.0, run, None, self.measuring_mode, None, 0, byref(enumerator))
            else:
                return False
            return True
        except OSError as dw_err:
            if dw_err.winerror != -2003283956:
                raise dw_err
        return False

    def render_single_glyph(self, font_face, indice, advance, offset, metrics):
        """Renders a single glyph using D2D DrawGlyphRun"""
        glyph_width, glyph_height, glyph_lsb, glyph_advance, glyph_bsb = metrics
        new_indice = (UINT16 * 1)(indice)
        new_advance = (FLOAT * 1)(advance)
        run = self._get_single_glyph_run(font_face, self.font._real_size, new_indice, new_advance, pointer(offset), False, 0)
        if self.draw_options & D2D1_DRAW_TEXT_OPTIONS_ENABLE_COLOR_FONT and self.is_color_run(run):
            return None
        if glyph_advance:
            render_width = int(math.ceil(glyph_advance * self.font.font_scale_ratio))
        else:
            render_width = int(math.ceil(glyph_width * self.font.font_scale_ratio))
        render_offset_x = 0
        if glyph_lsb < 0:
            render_offset_x = glyph_lsb * self.font.font_scale_ratio
        if self.font.italic:
            render_width += render_width // 2
        self._create_bitmap(render_width + 1, int(math.ceil(self.font.max_glyph_height)))
        baseline_offset = D2D_POINT_2F(-render_offset_x - offset.advanceOffset, self.font.ascent + offset.ascenderOffset)
        self._render_target.BeginDraw()
        self._render_target.Clear(transparent)
        self._render_target.DrawGlyphRun(baseline_offset, run, self._brush, self.measuring_mode)
        self._render_target.EndDraw(None, None)
        image = wic_decoder.get_image(self._bitmap)
        glyph = self.font.create_glyph(image)
        glyph.set_bearings(-self.font.descent, render_offset_x, advance, offset.advanceOffset, offset.ascenderOffset)
        return glyph

    def render_using_layout(self, text):
        """This will render text given the built in DirectWrite layout. This process allows us to take
        advantage of color glyphs and fallback handling that is built into DirectWrite.
        This can also handle shaping and many other features if you want to render directly to a texture."""
        text_layout = self.font.create_text_layout(text)
        layout_metrics = DWRITE_TEXT_METRICS()
        text_layout.GetMetrics(byref(layout_metrics))
        width = int(math.ceil(layout_metrics.width))
        height = int(math.ceil(layout_metrics.height))
        if width == 0 or height == 0:
            return None
        self._create_bitmap(width, height)
        point = D2D_POINT_2F(0, 0)
        self._render_target.BeginDraw()
        self._render_target.Clear(transparent)
        self._render_target.DrawTextLayout(point, text_layout, self._brush, self.draw_options)
        self._render_target.EndDraw(None, None)
        image = wic_decoder.get_image(self._bitmap)
        glyph = self.font.create_glyph(image)
        glyph.set_bearings(-self.font.descent, 0, int(math.ceil(layout_metrics.width)))
        return glyph

    def create_zero_glyph(self):
        """Zero glyph is a 1x1 image that has a -1 advance. This is to fill in for ligature substitutions since
        font system requires 1 glyph per character in a string."""
        self._create_bitmap(1, 1)
        image = wic_decoder.get_image(self._bitmap)
        glyph = self.font.create_glyph(image)
        glyph.set_bearings(-self.font.descent, 0, -1)
        return glyph

    def _create_bitmap(self, width, height):
        """Creates a bitmap using Direct2D and WIC."""
        if self._bitmap_dimensions[0] != width or self._bitmap_dimensions[1] != height:
            if self._bitmap:
                self._bitmap.Release()
            self._bitmap = IWICBitmap()
            wic_decoder._factory.CreateBitmap(width, height, GUID_WICPixelFormat32bppPBGRA, WICBitmapCacheOnDemand, byref(self._bitmap))
            self._render_target = ID2D1RenderTarget()
            d2d_factory.CreateWicBitmapRenderTarget(self._bitmap, default_target_properties, byref(self._render_target))
            self._render_target.SetTextAntialiasMode(self.antialias_mode)
            if not self._brush:
                self._brush = ID2D1SolidColorBrush()
                self._render_target.CreateSolidColorBrush(white, None, byref(self._brush))