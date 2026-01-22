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
def _assert_compile_called(self, method_name=None):
    if not self.compiled:
        msg = 'You must call `compile()` before '
        if metrics_module:
            msg += 'using the model.'
        else:
            msg += f'calling `{method_name}()`.'
        raise ValueError(msg)