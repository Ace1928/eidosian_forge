import collections
import copy
import os
from tensorflow.python.eager import def_function
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def try_build_compiled_arguments(model):
    if not version_utils.is_v1_layer_or_model(model) and model.outputs is not None:
        try:
            if not model.compiled_loss.built:
                model.compiled_loss.build(model.outputs)
            if not model.compiled_metrics.built:
                model.compiled_metrics.build(model.outputs, model.outputs)
        except:
            logging.warning('Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.')