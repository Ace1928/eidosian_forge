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
def to_mask(self, canal=0):
    """Return a mask a video clip made from the clip."""
    if self.ismask:
        return self
    else:
        newclip = self.fl_image(lambda pic: 1.0 * pic[:, :, canal] / 255)
        newclip.ismask = True
        return newclip