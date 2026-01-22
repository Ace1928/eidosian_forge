from __future__ import annotations
import os
from base64 import b64encode
from io import BytesIO
from typing import (
import numpy as np
import param
from ..models import Audio as _BkAudio, Video as _BkVideo
from ..util import isfile, isurl
from .base import ModelPane
class TensorLike(metaclass=TensorLikeMeta):
    """A class similar to torch.Tensor. We don't want to make PyTorch a dependency of this project
    """