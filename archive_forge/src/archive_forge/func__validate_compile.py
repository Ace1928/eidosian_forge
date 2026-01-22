import copy
import itertools
import json
import os
import warnings
import weakref
from tensorflow.python.autograph.lang import directives
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer as lso
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.python.keras.saving import save
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import constants as sm_constants
from tensorflow.python.saved_model import loader_impl as sm_loader
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.tools.docs import doc_controls
def _validate_compile(self, optimizer, metrics, **kwargs):
    """Performs validation checks for the default `compile`."""
    if any((isinstance(opt, optimizer_v1.Optimizer) for opt in nest.flatten(optimizer))):
        raise ValueError('`tf.compat.v1.keras` Optimizer (', optimizer, ') is not supported when eager execution is enabled. Use a `tf.keras` Optimizer instead, or disable eager execution.')
    kwargs.pop('cloning', None)
    kwargs.pop('experimental_run_tf_function', None)
    if kwargs.pop('distribute', None) is not None:
        raise ValueError('Distribute argument in compile is not available in TF 2.0 please create the model under the distribution strategy scope.')
    if kwargs.pop('target_tensors', None) is not None:
        raise ValueError('target_tensors argument is not supported when executing eagerly.')
    invalid_kwargs = set(kwargs) - {'sample_weight_mode'}
    if invalid_kwargs:
        raise TypeError('Invalid keyword argument(s) in `compile`: %s' % (invalid_kwargs,))
    if self.built and distribute_lib.has_strategy():
        strategy = distribute_lib.get_strategy()
        for v in self.variables:
            if not strategy.extended.variable_created_in_scope(v):
                raise ValueError('Variable (%s) was not created in the distribution strategy scope of (%s). It is most likely due to not all layers or the model or optimizer being created outside the distribution strategy scope. Try to make sure your code looks similar to the following.\nwith strategy.scope():\n  model=_create_model()\n  model.compile(...)' % (v, strategy))
    strategy = self.distribute_strategy
    for metric in nest.flatten(metrics):
        for v in getattr(metric, 'variables', []):
            if not strategy.extended.variable_created_in_scope(v):
                raise ValueError('Metric (%s) passed to model.compile was created inside of a different distribution strategy scope than the model. All metrics must be created in the same distribution strategy scope as the model (in this case %s). If you pass in a string identifier for a metric to compile the metric will automatically be created in the correct distribution strategy scope.' % (metric, strategy))
    for opt in nest.flatten(optimizer):
        for v in getattr(opt, '_weights', []):
            if not strategy.extended.variable_created_in_scope(v):
                raise ValueError('Optimizer (%s) passed to model.compile was created inside of a different distribution strategy scope than the model. All optimizers must be created in the same distribution strategy scope as the model (in this case %s). If you pass in a string identifier for an optimizer to compile the optimizer will automatically be created in the correct distribution strategy scope.' % (opt, strategy))