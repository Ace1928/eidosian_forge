import math
import numpy as np
import tree
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.utils.dataset_utils import is_torch_tensor
class ConverterIterableDataset(torch_data.IterableDataset):

    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        for batch in self.iterable:
            yield tree.map_structure(convert_to_tensor, batch)