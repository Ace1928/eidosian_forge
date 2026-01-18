import logging
import os
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Type, Union
from wandb import util
from wandb.sdk.lib import filesystem, runid
from . import _dtypes
from ._private import MEDIA_TMP
from .base_types.media import BatchableMedia
def write_gif_with_image_io(clip: Any, filename: str, fps: Optional[int]=None) -> None:
    imageio = util.get_module('imageio', required='wandb.Video requires imageio when passing raw data. Install with "pip install imageio"')
    writer = imageio.save(filename, fps=clip.fps, quantizer=0, palettesize=256, loop=0)
    for frame in clip.iter_frames(fps=fps, dtype='uint8'):
        writer.append_data(frame)
    writer.close()