import collections
import contextlib
import os
import re
import warnings
import numpy as np
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import distribution_lib
from keras.src.backend.common import global_state
def _maybe_populate_device_mesh(self, layout):
    if layout.device_mesh is None and self.device_mesh is not None:
        layout.device_mesh = self.device_mesh