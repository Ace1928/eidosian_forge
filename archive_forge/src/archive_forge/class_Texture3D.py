import re
import weakref
from ctypes import *
from io import open, BytesIO
import pyglet
from pyglet.gl import *
from pyglet.gl import gl_info
from pyglet.util import asbytes
from .codecs import ImageEncodeException, ImageDecodeException
from .codecs import registry as _codec_registry
from .codecs import add_default_codecs as _add_default_codecs
from .animation import Animation, AnimationFrame
from .buffer import *
from . import atlas
class Texture3D(Texture, UniformTextureSequence):
    """A texture with more than one image slice.

    Use `create_for_images` or `create_for_image_grid` classmethod to
    construct.
    """
    item_width = 0
    item_height = 0
    items = ()

    @classmethod
    def create_for_images(cls, images, internalformat=GL_RGBA, blank_data=True):
        item_width = images[0].width
        item_height = images[0].height
        for image in images:
            if image.width != item_width or image.height != item_height:
                raise ImageException('Images do not have same dimensions.')
        depth = len(images)
        texture = cls.create(item_width, item_height, GL_TEXTURE_3D, None)
        if images[0].anchor_x or images[0].anchor_y:
            texture.anchor_x = images[0].anchor_x
            texture.anchor_y = images[0].anchor_y
        texture.images = depth
        blank = (GLubyte * (texture.width * texture.height * texture.images))() if blank_data else None
        glBindTexture(texture.target, texture.id)
        glTexImage3D(texture.target, texture.level, internalformat, texture.width, texture.height, texture.images, 0, GL_ALPHA, GL_UNSIGNED_BYTE, blank)
        items = []
        for i, image in enumerate(images):
            item = cls.region_class(0, 0, i, item_width, item_height, texture)
            items.append(item)
            image.blit_to_texture(texture.target, texture.level, image.anchor_x, image.anchor_y, i)
        glFlush()
        texture.items = items
        texture.item_width = item_width
        texture.item_height = item_height
        return texture

    @classmethod
    def create_for_image_grid(cls, grid, internalformat=GL_RGBA):
        return cls.create_for_images(grid[:], internalformat)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __setitem__(self, index, value):
        if type(index) is slice:
            glBindTexture(self.target, self.id)
            for item, image in zip(self[index], value):
                image.blit_to_texture(self.target, self.level, image.anchor_x, image.anchor_y, item.z)
        else:
            self.blit_into(value, value.anchor_x, value.anchor_y, self[index].z)

    def __iter__(self):
        return iter(self.items)