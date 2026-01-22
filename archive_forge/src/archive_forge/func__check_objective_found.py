import collections
import hashlib
import os
import random
import threading
import warnings
from datetime import datetime
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import objective as obj_module
from keras_tuner.src.engine import stateful
from keras_tuner.src.engine import trial as trial_module
def _check_objective_found(self, metrics):
    if isinstance(self.objective, obj_module.MultiObjective):
        objective_names = list(self.objective.name_to_direction.keys())
    else:
        objective_names = [self.objective.name]
    for metric_name in metrics.keys():
        if metric_name in objective_names:
            objective_names.remove(metric_name)
    if objective_names:
        raise ValueError(f'Objective value missing in metrics reported to the Oracle, expected: {objective_names}, found: {metrics.keys()}')