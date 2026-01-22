import collections
import statistics
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src import errors
from keras_tuner.src.backend import config
from keras_tuner.src.backend import keras
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import objective as obj_module
def _get_best_value_and_best_epoch_from_history(history, objective):
    epoch_metrics = collections.defaultdict(dict)
    for metric_name, epoch_values in history.history.items():
        for epoch, value in enumerate(epoch_values):
            epoch_metrics[epoch][metric_name] = value
    best_epoch = 0
    for epoch, metrics in epoch_metrics.items():
        objective_value = objective.get_value(metrics)
        if objective.name not in metrics:
            metrics[objective.name] = objective_value
        best_value = epoch_metrics[best_epoch][objective.name]
        if objective.better_than(objective_value, best_value):
            best_epoch = epoch
    return (epoch_metrics[best_epoch], best_epoch)