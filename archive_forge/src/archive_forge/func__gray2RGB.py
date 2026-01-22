import subprocess as sp
import numpy as np
from .abstract import VideoReaderAbstract, VideoWriterAbstract
from .avprobe import avprobe
from .. import _AVCONV_APPLICATION
from .. import _AVCONV_PATH
from .. import _HAS_AVCONV
from ..utils import *
def _gray2RGB(self, data):
    T, M, N, C = data.shape
    if C < 3:
        vid = np.empty((T, M, N, C + 2), dtype=data.dtype)
        vid[:, :, :, 0] = data[:, :, :, 0]
        vid[:, :, :, 1] = data[:, :, :, 0]
        vid[:, :, :, 2] = data[:, :, :, 0]
        if C == 2:
            vid[:, :, :, 3] = data[:, :, :, 1]
        return vid
    return data