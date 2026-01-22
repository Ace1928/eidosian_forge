from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Pattern, Union, Optional, List, Any, Tuple, Callable, Iterator, Type, Dict, \
import pyglet
from pyglet import graphics
from pyglet.customtypes import AnchorX, AnchorY, ContentVAlign, HorizontalAlign
from pyglet.font.base import Font, Glyph
from pyglet.gl import GL_TRIANGLES, GL_LINES, glActiveTexture, GL_TEXTURE0, glBindTexture, glEnable, GL_BLEND, \
from pyglet.image import Texture
from pyglet.text import runlist
from pyglet.text.runlist import RunIterator, AbstractRunIterator
class _GlyphBox(_AbstractBox):
    owner: Texture
    font: Font
    glyphs: List[Tuple[int, Glyph]]
    advance: int
    vertex_lists: List[_LayoutVertexList]

    def __init__(self, owner: Texture, font: Font, glyphs: List[Tuple[int, Glyph]], advance: int) -> None:
        """Create a run of glyphs sharing the same texture.

        :Parameters:
            `owner` : `pyglet.image.Texture`
                Texture of all glyphs in this run.
            `font` : `pyglet.font.base.Font`
                Font of all glyphs in this run.
            `glyphs` : list of (int, `pyglet.font.base.Glyph`)
                Pairs of ``(kern, glyph)``, where ``kern`` gives horizontal
                displacement of the glyph in pixels (typically 0).
            `advance` : int
                Width of glyph run; must correspond to the sum of advances
                and kerns in the glyph list.

        """
        super().__init__(font.ascent, font.descent, advance, len(glyphs))
        assert owner
        self.owner = owner
        self.font = font
        self.glyphs = glyphs
        self.advance = advance
        self.vertex_lists = []

    def _add_vertex_list(self, vertex_list: Union[_LayoutVertexList, VertexList], context: _LayoutContext):
        self.vertex_lists.append(vertex_list)
        context.add_list(vertex_list)

    def place(self, layout: TextLayout, i: int, x: float, y: float, z: float, line_x: float, line_y: float, rotation: float, visible: bool, anchor_x: float, anchor_y: float, context: _LayoutContext) -> None:
        assert self.glyphs
        assert not self.vertex_lists
        try:
            group = layout.group_cache[self.owner]
        except KeyError:
            group = layout.group_class(self.owner, layout.program, order=1, parent=layout.group)
            layout.group_cache[self.owner] = group
        n_glyphs = self.length
        vertices = []
        tex_coords = []
        baseline = 0
        x1 = line_x
        for start, end, baseline in context.baseline_iter.ranges(i, i + n_glyphs):
            baseline = layout.parse_distance(baseline)
            assert len(self.glyphs[start - i:end - i]) == end - start
            for kern, glyph in self.glyphs[start - i:end - i]:
                x1 += kern
                v0, v1, v2, v3 = glyph.vertices
                v0 += x1
                v2 += x1
                v1 += line_y + baseline
                v3 += line_y + baseline
                vertices.extend(map(round, [v0, v1, 0, v2, v1, 0, v2, v3, 0, v0, v3, 0]))
                t = glyph.tex_coords
                tex_coords.extend(t)
                x1 += glyph.advance
        colors = []
        for start, end, color in context.colors_iter.ranges(i, i + n_glyphs):
            if color is None:
                color = (0, 0, 0, 255)
            if len(color) != 4:
                raise ValueError('Color requires 4 values (R, G, B, A). Value received: {}'.format(color))
            colors.extend(color * ((end - start) * 4))
        indices = []
        for glyph_idx in range(n_glyphs):
            indices.extend([element + glyph_idx * 4 for element in [0, 1, 2, 0, 2, 3]])
        t_position = (x, y, z)
        vertex_list = layout.program.vertex_list_indexed(n_glyphs * 4, GL_TRIANGLES, indices, layout.batch, group, position=('f', vertices), translation=('f', t_position * 4 * n_glyphs), colors=('Bn', colors), tex_coords=('f', tex_coords), rotation=('f', (rotation,) * 4 * n_glyphs), visible=('f', (visible,) * 4 * n_glyphs), anchor=('f', (anchor_x, anchor_y) * 4 * n_glyphs))
        self._add_vertex_list(vertex_list, context)
        background_vertices = []
        background_colors = []
        underline_vertices = []
        underline_colors = []
        y1 = line_y + self.descent + baseline
        y2 = line_y + self.ascent + baseline
        x1 = line_x
        for start, end, decoration in context.decoration_iter.ranges(i, i + n_glyphs):
            bg, underline = decoration
            x2 = x1
            for kern, glyph in self.glyphs[start - i:end - i]:
                x2 += glyph.advance + kern
            if bg is not None:
                if len(bg) != 4:
                    raise ValueError(f'Background color requires 4 values (R, G, B, A). Value received: {bg}')
                background_vertices.extend([x1, y1, 0, x2, y1, 0, x2, y2, 0, x1, y2, 0])
                background_colors.extend(bg * 4)
            if underline is not None:
                if len(underline) != 4:
                    raise ValueError(f'Underline color requires 4 values (R, G, B, A). Value received: {underline}')
                underline_vertices.extend([x1, line_y + baseline - 2, 0, x2, line_y + baseline - 2, 0])
                underline_colors.extend(underline * 2)
            x1 = x2
        if background_vertices:
            bg_count = len(background_vertices) // 3
            background_indices = [(0, 1, 2, 0, 2, 3)[i % 6] for i in range(bg_count * 3)]
            decoration_program = get_default_decoration_shader()
            background_list = decoration_program.vertex_list_indexed(bg_count, GL_TRIANGLES, background_indices, layout.batch, layout.background_decoration_group, position=('f', background_vertices), translation=('f', t_position * bg_count), colors=('Bn', background_colors), rotation=('f', (rotation,) * bg_count), visible=('f', (visible,) * bg_count), anchor=('f', (anchor_x, anchor_y) * bg_count))
            self._add_vertex_list(background_list, context)
        if underline_vertices:
            ul_count = len(underline_vertices) // 3
            decoration_program = get_default_decoration_shader()
            underline_list = decoration_program.vertex_list(ul_count, GL_LINES, layout.batch, layout.foreground_decoration_group, position=('f', underline_vertices), translation=('f', t_position * ul_count), colors=('Bn', underline_colors), rotation=('f', (rotation,) * ul_count), visible=('f', (visible,) * ul_count), anchor=('f', (anchor_x, anchor_y) * ul_count))
            self._add_vertex_list(underline_list, context)

    def update_translation(self, x: float, y: float, z: float) -> None:
        translation = (x, y, z)
        for _vertex_list in self.vertex_lists:
            _vertex_list.translation[:] = translation * _vertex_list.count

    def update_colors(self, colors: List[int]) -> None:
        for _vertex_list in self.vertex_lists:
            _vertex_list.colors[:] = colors[:_vertex_list.count] * 4

    def update_view_translation(self, translate_x: float, translate_y: float) -> None:
        view_translation = (-translate_x, -translate_y, 0)
        for _vertex_list in self.vertex_lists:
            _vertex_list.view_translation[:] = view_translation * _vertex_list.count

    def update_rotation(self, rotation: float) -> None:
        rot = (rotation,)
        for _vertex_list in self.vertex_lists:
            _vertex_list.rotation[:] = rot * _vertex_list.count

    def update_visibility(self, visible: bool) -> None:
        visible_tuple = (visible,)
        for _vertex_list in self.vertex_lists:
            _vertex_list.visible[:] = visible_tuple * _vertex_list.count

    def update_anchor(self, anchor_x: float, anchor_y: float) -> None:
        anchor = (anchor_x, anchor_y)
        for _vertex_list in self.vertex_lists:
            _vertex_list.anchor[:] = anchor * _vertex_list.count

    def delete(self, layout: TextLayout) -> None:
        for _vertex_list in self.vertex_lists:
            _vertex_list.delete()
        self.vertex_lists.clear()

    def get_point_in_box(self, position: int) -> int:
        x = 0
        for kern, glyph in self.glyphs:
            if position == 0:
                break
            position -= 1
            x += glyph.advance + kern
        return x

    def get_position_in_box(self, x: int) -> int:
        position = 0
        last_glyph_x = 0
        for kern, glyph in self.glyphs:
            last_glyph_x += kern
            if last_glyph_x + glyph.advance // 2 > x:
                return position
            position += 1
            last_glyph_x += glyph.advance
        return position

    def __repr__(self) -> str:
        return f'_GlyphBox({self.glyphs})'