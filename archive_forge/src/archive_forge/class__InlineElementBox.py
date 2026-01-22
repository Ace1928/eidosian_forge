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
class _InlineElementBox(_AbstractBox):
    element: InlineElement
    placed: bool

    def __init__(self, element: InlineElement) -> None:
        """Create a glyph run holding a single element."""
        super().__init__(element.ascent, element.descent, element.advance, 1)
        self.element = element
        self.placed = False

    def place(self, layout: TextLayout, i: int, x: float, y: float, z: float, line_x: float, line_y: float, rotation: float, visible: bool, anchor_x: float, anchor_y: float, context: _LayoutContext) -> None:
        self.element.place(layout, x, y, z, line_x, line_y, rotation, visible, anchor_x, anchor_y)
        self.placed = True

    def update_translation(self, x: float, y: float, z: float) -> None:
        if self.placed:
            self.element.update_translation(x, y, z)

    def update_colors(self, colors: List[int]) -> None:
        if self.placed:
            self.element.update_color(colors)

    def update_view_translation(self, translate_x: float, translate_y: float) -> None:
        if self.placed:
            self.element.update_view_translation(translate_x, translate_y)

    def update_rotation(self, rotation: float) -> None:
        if self.placed:
            self.element.update_rotation(rotation)

    def update_visibility(self, visible: bool) -> None:
        if self.placed:
            self.element.update_visibility(visible)

    def update_anchor(self, anchor_x: float, anchor_y: float) -> None:
        if self.placed:
            self.element.update_anchor(anchor_x, anchor_y)

    def delete(self, layout: TextLayout) -> None:
        if self.placed:
            self.element.remove(layout)
            self.placed = False

    def get_point_in_box(self, position: int) -> float:
        if position == 0:
            return 0
        else:
            return self.advance

    def get_position_in_box(self, x: int) -> int:
        if x < self.advance // 2:
            return 0
        else:
            return 1

    def __repr__(self) -> str:
        return f'_InlineElementBox({self.element})'