import os
import time
import warnings
import numpy as np
from .. import _HAS_FFMPEG
from ..utils import *
def writeFrame(self, im):
    """Sends ndarray frames to FFmpeg

        """
    vid = vshape(im)
    T, M, N, C = vid.shape
    if not self.warmStarted:
        self._warmStart(M, N, C, im.dtype)
    vid = vid.clip(0, (1 << (self.dtype.itemsize << 3)) - 1).astype(self.dtype)
    vid = self._prepareData(vid)
    T, M, N, C = vid.shape
    if self.inputdict['-pix_fmt'].startswith('yuv444p') or self.inputdict['-pix_fmt'].startswith('yuvj444p') or self.inputdict['-pix_fmt'].startswith('yuva444p'):
        vid = vid.transpose((0, 3, 1, 2))
    if M != self.inputheight or N != self.inputwidth:
        raise ValueError('All images in a movie should have same size')
    if C != self.inputNumChannels:
        raise ValueError('All images in a movie should have same number of channels')
    assert self._proc is not None
    try:
        self._proc.stdin.write(vid.tostring())
    except IOError as e:
        msg = '{0:}\n\nFFMPEG COMMAND:\n{1:}\n\nFFMPEG STDERR OUTPUT:\n'.format(e, self._cmd)
        raise IOError(msg)