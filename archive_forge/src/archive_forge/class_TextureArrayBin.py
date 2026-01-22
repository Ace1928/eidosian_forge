from typing import TYPE_CHECKING, Tuple, Optional
import pyglet
class TextureArrayBin:
    """Collection of texture arrays.

    :py:class:`~pyglet.image.atlas.TextureArrayBin` maintains a collection of texture arrays, and creates new
    ones as necessary as the depth is exceeded.
    """

    def __init__(self, texture_width: int=2048, texture_height: int=2048, max_depth: Optional[int]=None) -> None:
        max_texture_size = pyglet.image.get_max_texture_size()
        self.max_depth = max_depth or pyglet.image.get_max_array_texture_layers()
        self.texture_width = min(texture_width, max_texture_size)
        self.texture_height = min(texture_height, max_texture_size)
        self.arrays = []

    def add(self, img: 'AbstractImage') -> 'TextureArrayRegion':
        """Add an image into this texture array bin.

        This method calls `TextureArray.add` for the first array that has room
        for the image.

        `TextureArraySizeExceeded` is raised if the image exceeds the dimensions of
        ``texture_width`` and ``texture_height``.

        :Parameters:
            `img` : `~pyglet.image.AbstractImage`
                The image to add.

        :rtype: :py:class:`~pyglet.image.TextureArrayRegion`
        :return: The region of an array containing the newly added image.
        """
        try:
            array = self.arrays[-1]
            return array.add(img)
        except pyglet.image.TextureArrayDepthExceeded:
            pass
        except IndexError:
            pass
        array = pyglet.image.TextureArray.create(self.texture_width, self.texture_height, max_depth=self.max_depth)
        self.arrays.append(array)
        return array.add(img)