from __future__ import annotations
from typing import Type, Optional, TYPE_CHECKING, Tuple
from pyglet import graphics
from pyglet.customtypes import AnchorY, AnchorX
from pyglet.gl import glActiveTexture, GL_TEXTURE0, glBindTexture, glEnable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, \
from pyglet.text.layout.base import TextLayout
class ScrollableTextLayoutGroup(graphics.Group):
    scissor_area = (0, 0, 0, 0)

    def __init__(self, texture: Texture, program: ShaderProgram, order: int=1, parent: Optional[graphics.Group]=None) -> None:
        """Default rendering group for :py:class:`~pyglet.text.layout.ScrollableTextLayout`.

        The group maintains internal state for specifying the viewable
        area, and for scrolling. Because the group has internal state
        specific to the text layout, the group is never shared.
        """
        super().__init__(order=order, parent=parent)
        self.texture = texture
        self.program = program

    def set_state(self) -> None:
        self.program.use()
        self.program['scissor'] = True
        self.program['scissor_area'] = self.scissor_area
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(self.texture.target, self.texture.id)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def unset_state(self) -> None:
        glDisable(GL_BLEND)
        self.program.stop()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.texture})'

    def __eq__(self, other: graphics.Group) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)