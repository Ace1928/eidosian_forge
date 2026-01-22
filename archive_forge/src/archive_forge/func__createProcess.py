import subprocess as sp
import numpy as np
from .abstract import VideoReaderAbstract, VideoWriterAbstract
from .ffprobe import ffprobe
from .. import _FFMPEG_APPLICATION
from .. import _FFMPEG_PATH
from .. import _FFMPEG_SUPPORTED_DECODERS
from .. import _FFMPEG_SUPPORTED_ENCODERS
from .. import _HAS_FFMPEG
from ..utils import *
def _createProcess(self, inputdict, outputdict, verbosity):
    iargs = self._dict2Args(inputdict)
    oargs = self._dict2Args(outputdict)
    cmd = [_FFMPEG_PATH + '/' + _FFMPEG_APPLICATION, '-y'] + iargs + ['-i', '-'] + oargs + [self._filename]
    self._cmd = ' '.join(cmd)
    if self.verbosity > 0:
        print(self._cmd)
        self._proc = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=None)
    else:
        self._proc = sp.Popen(cmd, stdin=sp.PIPE, stdout=self.DEVNULL, stderr=sp.STDOUT)