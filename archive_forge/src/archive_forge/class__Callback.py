import shutil
from typing import Dict, List, Optional, Union
from tensorflow.keras.callbacks import Callback as KerasCallback
import ray
from ray.train.tensorflow import TensorflowCheckpoint
from ray.util.annotations import PublicAPI
class _Callback(KerasCallback):
    """Base class for Air's Keras callbacks."""
    _allowed = ['epoch_begin', 'epoch_end', 'train_batch_begin', 'train_batch_end', 'test_batch_begin', 'test_batch_end', 'predict_batch_begin', 'predict_batch_end', 'train_begin', 'train_end', 'test_begin', 'test_end', 'predict_begin', 'predict_end']

    def __init__(self, on: Union[str, List[str]]='validation_end'):
        super(_Callback, self).__init__()
        if not isinstance(on, list):
            on = [on]
        if any((w not in self._allowed for w in on)):
            raise ValueError('Invalid trigger time selected: {}. Must be one of {}'.format(on, self._allowed))
        self._on = on

    def _handle(self, logs: Dict, when: str):
        raise NotImplementedError

    def on_epoch_begin(self, epoch, logs=None):
        if 'epoch_begin' in self._on:
            self._handle(logs, 'epoch_begin')

    def on_epoch_end(self, epoch, logs=None):
        if 'epoch_end' in self._on:
            self._handle(logs, 'epoch_end')

    def on_train_batch_begin(self, batch, logs=None):
        if 'train_batch_begin' in self._on:
            self._handle(logs, 'train_batch_begin')

    def on_train_batch_end(self, batch, logs=None):
        if 'train_batch_end' in self._on:
            self._handle(logs, 'train_batch_end')

    def on_test_batch_begin(self, batch, logs=None):
        if 'test_batch_begin' in self._on:
            self._handle(logs, 'test_batch_begin')

    def on_test_batch_end(self, batch, logs=None):
        if 'test_batch_end' in self._on:
            self._handle(logs, 'test_batch_end')

    def on_predict_batch_begin(self, batch, logs=None):
        if 'predict_batch_begin' in self._on:
            self._handle(logs, 'predict_batch_begin')

    def on_predict_batch_end(self, batch, logs=None):
        if 'predict_batch_end' in self._on:
            self._handle(logs, 'predict_batch_end')

    def on_train_begin(self, logs=None):
        if 'train_begin' in self._on:
            self._handle(logs, 'train_begin')

    def on_train_end(self, logs=None):
        if 'train_end' in self._on:
            self._handle(logs, 'train_end')

    def on_test_begin(self, logs=None):
        if 'test_begin' in self._on:
            self._handle(logs, 'test_begin')

    def on_test_end(self, logs=None):
        if 'test_end' in self._on:
            self._handle(logs, 'test_end')

    def on_predict_begin(self, logs=None):
        if 'predict_begin' in self._on:
            self._handle(logs, 'predict_begin')

    def on_predict_end(self, logs=None):
        if 'predict_end' in self._on:
            self._handle(logs, 'predict_end')