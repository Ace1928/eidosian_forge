from __future__ import annotations
import re
from typing import List, TYPE_CHECKING, Optional, Any
import pyglet
import pyglet.text.layout
from pyglet.gl import GL_TEXTURE0, glActiveTexture, glBindTexture, glEnable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, \
class ImageElement(pyglet.text.document.InlineElement):
    height: int
    width: int

    def __init__(self, image: AbstractImage, width: Optional[int]=None, height: Optional[int]=None):
        self.image = image.get_texture()
        self.width = width is None and image.width or width
        self.height = height is None and image.height or height
        self.vertex_lists = {}
        anchor_y = self.height // image.height * image.anchor_y
        ascent = max(0, self.height - anchor_y)
        descent = min(0, -anchor_y)
        super().__init__(ascent, descent, self.width)

    def place(self, layout: TextLayout, x: float, y: float, z: float, line_x: float, line_y: float, rotation: float, visible: bool, anchor_x: float, anchor_y: float) -> None:
        program = pyglet.text.layout.get_default_image_layout_shader()
        group = _InlineElementGroup(self.image.get_texture(), program, 0, layout.group)
        x1 = line_x
        y1 = line_y + self.descent
        x2 = line_x + self.width
        y2 = line_y + self.height + self.descent
        vertex_list = program.vertex_list_indexed(4, pyglet.gl.GL_TRIANGLES, [0, 1, 2, 0, 2, 3], layout.batch, group, position=('f', (x1, y1, z, x2, y1, z, x2, y2, z, x1, y2, z)), translation=('f', (x, y, z) * 4), tex_coords=('f', self.image.tex_coords), visible=('f', (visible,) * 4), rotation=('f', (rotation,) * 4), anchor=('f', (anchor_x, anchor_y) * 4))
        self.vertex_lists[layout] = vertex_list

    def update_translation(self, x: float, y: float, z: float) -> None:
        translation = (x, y, z)
        for _vertex_list in self.vertex_lists.values():
            _vertex_list.translation[:] = translation * _vertex_list.count

    def update_color(self, color: List[int]) -> None:
        ...

    def update_view_translation(self, translate_x: float, translate_y: float) -> None:
        view_translation = (-translate_x, -translate_y, 0)
        for _vertex_list in self.vertex_lists.values():
            _vertex_list.view_translation[:] = view_translation * _vertex_list.count

    def update_rotation(self, rotation: float) -> None:
        rot_tuple = (rotation,)
        for _vertex_list in self.vertex_lists.values():
            _vertex_list.rotation[:] = rot_tuple * _vertex_list.count

    def update_visibility(self, visible: bool) -> None:
        visible_tuple = (visible,)
        for _vertex_list in self.vertex_lists.values():
            _vertex_list.visible[:] = visible_tuple * _vertex_list.count

    def update_anchor(self, anchor_x: float, anchor_y: float) -> None:
        anchor = (anchor_x, anchor_y)
        for _vertex_list in self.vertex_lists.values():
            _vertex_list.anchor[:] = anchor * _vertex_list.count

    def remove(self, layout: TextLayout) -> None:
        self.vertex_lists[layout].delete()
        del self.vertex_lists[layout]