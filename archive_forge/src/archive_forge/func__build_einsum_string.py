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
def _build_einsum_string(self, free_input_dims, bound_dims, output_dims):
    _CHR_IDX = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']
    input_str = ''
    kernel_str = ''
    output_str = ''
    letter_offset = 0
    for i in range(free_input_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        output_str += char
    letter_offset += free_input_dims
    for i in range(bound_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        kernel_str += char
    letter_offset += bound_dims
    for i in range(output_dims):
        char = _CHR_IDX[i + letter_offset]
        kernel_str += char
        output_str += char
    return input_str + ',' + kernel_str + '->' + output_str