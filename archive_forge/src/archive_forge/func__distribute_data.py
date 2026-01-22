import collections
import itertools
from functools import partial
import jax
import numpy as np
import tree
from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import ops
from keras.src import optimizers as optimizers_module
from keras.src.backend import distribution_lib as jax_distribution_lib
from keras.src.distribution import distribution_lib
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
def _distribute_data(data):
    distribution = distribution_lib.distribution()
    if distribution is not None:

        def distribute_single_value(d):
            layout = distribution.get_data_layout(d.shape)
            return jax_distribution_lib.distribute_data_input(d, layout)
        return tree.map_structure(distribute_single_value, data)
    else:
        return tree.map_structure(jax.device_put, data)