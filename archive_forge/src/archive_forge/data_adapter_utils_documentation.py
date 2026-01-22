import math
import numpy as np
import tree
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.utils.dataset_utils import is_torch_tensor
Split arrays into train and validation subsets in deterministic order.

    The last part of data will become validation data.

    Args:
        arrays: Tensors to split. Allowed inputs are arbitrarily nested
            structures of Tensors and NumPy arrays.
        validation_split: Float between 0 and 1. The proportion of the dataset
            to include in the validation split. The rest of the dataset will be
            included in the training split.

    Returns:
        `(train_arrays, validation_arrays)`
    