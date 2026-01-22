from contextlib import ExitStack
from functools import partial
import math
import numpy as np
import warnings
from affine import Affine
from rasterio.env import env_ctx_if_needed
from rasterio._transform import (
from rasterio.enums import TransformDirection, TransformMethod
from rasterio.control import GroundControlPoint
from rasterio.rpc import RPC
from rasterio.errors import TransformError, RasterioDeprecationWarning
class GDALTransformerBase(TransformerBase):

    def __init__(self):
        super().__init__()
        self._env = ExitStack()

    def close(self):
        pass

    def __enter__(self):
        self._env.enter_context(env_ctx_if_needed())
        return self

    def __exit__(self, *args):
        self.close()
        self._env.close()