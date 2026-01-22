import json
import threading
import tree
from absl import logging
from keras.src import backend
from keras.src import layers
from keras.src import losses
from keras.src import metrics as metrics_module
from keras.src import models
from keras.src import optimizers
from keras.src.legacy.saving import serialization
from keras.src.saving import object_registration
def _find_replace_nested_dict(config, find, replace):
    dict_str = json.dumps(config)
    dict_str = dict_str.replace(find, replace)
    config = json.loads(dict_str)
    return config