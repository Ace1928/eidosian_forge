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
@requires_duration
@use_clip_fps_by_default
@convert_masks_to_RGB
def write_images_sequence(self, nameformat, fps=None, verbose=True, withmask=True, logger='bar'):
    """ Writes the videoclip to a sequence of image files.

        Parameters
        -----------

        nameformat
          A filename specifying the numerotation format and extension
          of the pictures. For instance "frame%03d.png" for filenames
          indexed with 3 digits and PNG format. Also possible:
          "some_folder/frame%04d.jpeg", etc.

        fps
          Number of frames per second to consider when writing the
          clip. If not specified, the clip's ``fps`` attribute will
          be used if it has one.

        withmask
          will save the clip's mask (if any) as an alpha canal (PNGs only).

        verbose
          Boolean indicating whether to print information.

        logger
          Either 'bar' (progress bar) or None or any Proglog logger.


        Returns
        --------

        names_list
          A list of all the files generated.

        Notes
        ------

        The resulting image sequence can be read using e.g. the class
        ``ImageSequenceClip``.

        """
    logger = proglog.default_bar_logger(logger)
    logger(message='Moviepy - Writing frames %s.' % nameformat)
    tt = np.arange(0, self.duration, 1.0 / fps)
    filenames = []
    for i, t in logger.iter_bar(t=list(enumerate(tt))):
        name = nameformat % i
        filenames.append(name)
        self.save_frame(name, t, withmask=withmask)
    logger(message='Moviepy - Done writing frames %s.' % nameformat)
    return filenames