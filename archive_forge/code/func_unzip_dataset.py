import numpy as np
import tensorflow as tf
from tensorflow import nest
def unzip_dataset(dataset):
    return nest.flatten([dataset.map(lambda *a: nest.flatten(a)[index]) for index in range(len(nest.flatten(dataset_shape(dataset))))])