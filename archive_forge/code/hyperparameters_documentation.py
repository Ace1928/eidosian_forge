import collections
import contextlib
import copy
import six
from keras_tuner.src import protos
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import conditions as conditions_mod
from keras_tuner.src.engine.hyperparameters import hp_types
from keras_tuner.src.engine.hyperparameters import hyperparameter as hp_module
Returns a name qualified by `name_scopes`.