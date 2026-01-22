import warnings
import numpy as np
import torch
import tree
from packaging.version import parse
from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import optimizers as optimizers_module
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
class TorchEpochIterator(EpochIterator):

    def _get_iterator(self):
        return self.data_adapter.get_torch_dataloader()