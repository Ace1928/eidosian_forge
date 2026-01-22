import os
import subprocess as sp
import tempfile
import warnings
import numpy as np
import proglog
from imageio import imread, imsave
from ..Clip import Clip
from ..compat import DEVNULL, string_types
from ..config import get_setting
from ..decorators import (add_mask_if_none, apply_to_mask,
from ..tools import (deprecated_version_of, extensions_dict, find_extension,
from .io.ffmpeg_writer import ffmpeg_write_video
from .io.gif_writers import (write_gif, write_gif_with_image_io,
from .tools.drawing import blit
class UpdatedVideoClip(VideoClip):
    """
    Class of clips whose make_frame requires some objects to
    be updated. Particularly practical in science where some
    algorithm needs to make some steps before a new frame can
    be generated.

    UpdatedVideoClips have the following make_frame:

    >>> def make_frame(t):
    >>>     while self.world.clip_t < t:
    >>>         world.update() # updates, and increases world.clip_t
    >>>     return world.to_frame()

    Parameters
    -----------

    world
      An object with the following attributes:
      - world.clip_t : the clip's time corresponding to the
          world's state
      - world.update() : update the world's state, (including
        increasing world.clip_t of one time step)
      - world.to_frame() : renders a frame depending on the world's state

    ismask
      True if the clip is a WxH mask with values in 0-1

    duration
      Duration of the clip, in seconds

    """

    def __init__(self, world, ismask=False, duration=None):
        self.world = world

        def make_frame(t):
            while self.world.clip_t < t:
                world.update()
            return world.to_frame()
        VideoClip.__init__(self, make_frame=make_frame, ismask=ismask, duration=duration)