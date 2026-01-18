import numpy as np
from keras.src.api_export import keras_export
from keras.src.backend.config import standardize_data_format
from keras.src.utils import dataset_utils
from keras.src.utils import image_utils
from keras.src.utils.module_utils import tensorflow as tf
Load an image from a path and resize it.