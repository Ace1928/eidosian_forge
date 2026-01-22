import os
import time
import warnings
import numpy as np
from .. import _HAS_FFMPEG
from ..utils import *
def _readFrame(self):
    frame = self._read_frame_data()
    if len(frame) == 0:
        return frame
    if self.output_pix_fmt == 'rgb24':
        self._lastread = frame.reshape((self.outputheight, self.outputwidth, self.outputdepth))
    elif self.output_pix_fmt.startswith('yuv444p') or self.output_pix_fmt.startswith('yuvj444p') or self.output_pix_fmt.startswith('yuva444p'):
        self._lastread = frame.reshape((self.outputdepth, self.outputheight, self.outputwidth)).transpose((1, 2, 0))
    else:
        if self.verbosity > 0:
            warnings.warn('Unsupported reshaping from raw buffer to images frames  for format {:}. Assuming HEIGHTxWIDTHxCOLOR'.format(self.output_pix_fmt), UserWarning)
        self._lastread = frame.reshape((self.outputheight, self.outputwidth, self.outputdepth))
    return self._lastread