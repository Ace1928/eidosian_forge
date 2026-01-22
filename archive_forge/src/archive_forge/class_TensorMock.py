import pathlib
from base64 import b64encode
from io import BytesIO
import numpy as np
import pytest
from panel.pane import Audio, Video
from panel.pane.media import TensorLike, _is_1dim_int_or_float_tensor
class TensorMock:

    def __init__(self, duration=2.0):
        self._data = get_audio_np_float32(duration=duration)

    def numpy(self):
        return self._data

    def dim(self):
        return self._data.ndim

    @property
    def dtype(self):
        return 'torch.float32'