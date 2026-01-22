import platform
import warnings
import tree
from keras.src import backend
from keras.src import metrics as metrics_module
from keras.src import ops
from keras.src import optimizers
from keras.src.optimizers.loss_scale_optimizer import LossScaleOptimizer
from keras.src.saving import serialization_lib
from keras.src.trainers.compile_utils import CompileLoss
from keras.src.trainers.compile_utils import CompileMetrics
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils import traceback_utils
from keras.src.utils import tracking
def _flatten_metrics_in_order(self, logs):
    """Turns `logs` dict into a list as per key order of `metrics_names`."""
    metric_names = [m.name for m in self.metrics]
    results = []
    for name in metric_names:
        if name in logs:
            results.append(logs[name])
    for key in sorted(logs.keys()):
        if key not in metric_names:
            results.append(logs[key])
    if len(results) == 1:
        return results[0]
    return results