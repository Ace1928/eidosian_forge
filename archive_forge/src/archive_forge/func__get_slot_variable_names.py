from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf
from tensorflow.python.feature_column import feature_column as core_fc
from tensorflow.python.feature_column import feature_column_lib as core_fc_lib
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import ops
from tensorflow.python.tpu import feature_column as tpu_fc
from tensorflow.python.tpu import feature_column_v2 as tpu_fc_v2
from tensorflow.python.tpu import tpu_embedding
from tensorflow.python.tpu.tpu_embedding import AdagradParameters
from tensorflow.python.tpu.tpu_embedding import AdamParameters
from tensorflow.python.tpu.tpu_embedding import FtrlParameters
from tensorflow.python.tpu.tpu_embedding import MomentumParameters
from tensorflow.python.tpu.tpu_embedding import ProximalAdagradParameters
from tensorflow.python.tpu.tpu_embedding import RMSPropParameters
from tensorflow.python.tpu.tpu_embedding import StochasticGradientDescentParameters
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _get_slot_variable_names(scope_name, var_name, optimization_parameters):
    """Return embedding variable names which are consistent with CPU runs."""
    if scope_name:
        scope_name = scope_name + '/'
    if isinstance(optimization_parameters, tf.compat.v1.tpu.experimental.AdagradParameters):
        return tpu_embedding.AdagradSlotVariableNames('{}{}/Adagrad'.format(scope_name, var_name))
    elif isinstance(optimization_parameters, tf.compat.v1.tpu.experimental.AdamParameters):
        return tpu_embedding.AdamSlotVariableNames('{}{}/Adam/m'.format(scope_name, var_name), '{}{}/Adam/v'.format(scope_name, var_name))
    elif isinstance(optimization_parameters, tf.compat.v1.tpu.experimental.FtrlParameters):
        return tpu_embedding.FtrlSlotVariableNames('{}{}/Ftrl'.format(scope_name, var_name), '{}{}/Ftrl_1'.format(scope_name, var_name))
    elif isinstance(optimization_parameters, MomentumParameters):
        return tpu_embedding.MomentumSlotVariableNames('{}{}/Momentum'.format(scope_name, var_name))
    elif isinstance(optimization_parameters, RMSPropParameters):
        return tpu_embedding.RMSPropSlotVariableNames(ms='{}{}/RMSProp/ms'.format(scope_name, var_name), mom='{}{}/RMSProp/mom'.format(scope_name, var_name))
    elif isinstance(optimization_parameters, ProximalAdagradParameters):
        return tpu_embedding.ProximalAdagradSlotVariableNames('{}{}/ProximalAdagrad'.format(scope_name, var_name))
    elif isinstance(optimization_parameters, tf.compat.v1.tpu.experimental.StochasticGradientDescentParameters):
        return None
    else:
        raise ValueError('Support to infer full variable name for optimization_parameter {} has not been added.'.format(optimization_parameters))