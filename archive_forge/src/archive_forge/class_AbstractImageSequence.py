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
class AbstractImageSequence:
    """Abstract sequence of images.

    The sequence is useful for storing image animations or slices of a volume.
    For efficient access, use the `texture_sequence` member.  The class
    also implements the sequence interface (`__len__`, `__getitem__`,
    `__setitem__`).
    """

    def get_texture_sequence(self):
        """Get a TextureSequence.

        :rtype: `TextureSequence`

        .. versionadded:: 1.1
        """
        raise NotImplementedError('abstract')

    def get_animation(self, period, loop=True):
        """Create an animation over this image sequence for the given constant
        framerate.

        :Parameters
            `period` : float
                Number of seconds to display each frame.
            `loop` : bool
                If True, the animation will loop continuously.

        :rtype: :py:class:`~pyglet.image.Animation`

        .. versionadded:: 1.1
        """
        return Animation.from_image_sequence(self, period, loop)

    def __getitem__(self, slice):
        """Retrieve a (list of) image.

        :rtype: AbstractImage
        """
        raise NotImplementedError('abstract')

    def __setitem__(self, slice, image):
        """Replace one or more images in the sequence.

        :Parameters:
            `image` : `~pyglet.image.AbstractImage`
                The replacement image.  The actual instance may not be used,
                depending on this implementation.

        """
        raise NotImplementedError('abstract')

    def __len__(self):
        raise NotImplementedError('abstract')

    def __iter__(self):
        """Iterate over the images in sequence.

        :rtype: Iterator

        .. versionadded:: 1.1
        """
        raise NotImplementedError('abstract')