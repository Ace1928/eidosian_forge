import sys
import pyglet
from pyglet.gl import *
from pyglet import clock
from pyglet import event
from pyglet import graphics
from pyglet import image
class SpriteGroup(graphics.Group):
    """Shared sprite rendering group.

    The group is automatically coalesced with other sprite groups sharing the
    same parent group, texture and blend parameters.
    """

    def __init__(self, texture, blend_src, blend_dest, program, parent=None):
        """Create a sprite group.

        The group is created internally when a :py:class:`~pyglet.sprite.Sprite`
        is created; applications usually do not need to explicitly create it.

        :Parameters:
            `texture` : `~pyglet.image.Texture`
                The (top-level) texture containing the sprite image.
            `blend_src` : int
                OpenGL blend source mode; for example,
                ``GL_SRC_ALPHA``.
            `blend_dest` : int
                OpenGL blend destination mode; for example,
                ``GL_ONE_MINUS_SRC_ALPHA``.
            `program` : `~pyglet.graphics.shader.ShaderProgram`
                A custom ShaderProgram.
            `order` : int
                Change the order to render above or below other Groups.
            `parent` : `~pyglet.graphics.Group`
                Optional parent group.
        """
        super().__init__(parent=parent)
        self.texture = texture
        self.blend_src = blend_src
        self.blend_dest = blend_dest
        self.program = program

    def set_state(self):
        self.program.use()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(self.texture.target, self.texture.id)
        glEnable(GL_BLEND)
        glBlendFunc(self.blend_src, self.blend_dest)

    def unset_state(self):
        glDisable(GL_BLEND)
        self.program.stop()

    def __repr__(self):
        return '{0}({1})'.format(self.__class__.__name__, self.texture)

    def __eq__(self, other):
        return other.__class__ is self.__class__ and self.program is other.program and (self.parent == other.parent) and (self.texture.target == other.texture.target) and (self.texture.id == other.texture.id) and (self.blend_src == other.blend_src) and (self.blend_dest == other.blend_dest)

    def __hash__(self):
        return hash((self.program, self.parent, self.texture.id, self.texture.target, self.blend_src, self.blend_dest))