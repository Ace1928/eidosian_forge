from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import re
import tensorflow.compat.v2 as tf
from keras.src.engine.base_layer import Layer
from keras.src.saving import serialization_lib
Verifies and concatenates the dense output of several columns.