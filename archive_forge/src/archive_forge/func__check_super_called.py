import re
import warnings
import numpy as np
from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
from keras.src.utils.naming import auto_name
def _check_super_called(self):
    if not hasattr(self, '_lock'):
        raise RuntimeError(f"In optimizer '{self.__class__.__name__}', you forgot to call `super().__init__()` as the first statement in the `__init__()` method. Go add it!")