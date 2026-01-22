import collections
import math
import os
import re
import unicodedata
from typing import List
import numpy as np
import six
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from autokeras import constants
from autokeras.utils import data_utils
@keras.utils.register_keras_serializable()
class CastToFloat32(preprocessing.PreprocessingLayer):

    def get_config(self):
        return super().get_config()

    def call(self, inputs):
        return data_utils.cast_to_float32(inputs)

    def adapt(self, data):
        return