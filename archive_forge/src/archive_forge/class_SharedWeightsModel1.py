import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
class SharedWeightsModel1(TFModelV2):
    """Example of weight sharing between two different TFModelV2s.

    NOTE: This will only work for tf1 (static graph). When running with
    config.framework_str=tf2, use TF2SharedWeightsModel, instead!

    Here, we share the variables defined in the 'shared' variable scope
    by entering it explicitly with tf1.AUTO_REUSE. This creates the
    variables for the 'fc1' layer in a global scope called 'shared'
    (outside of the Policy's normal variable scope).
    """

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        super().__init__(observation_space, action_space, num_outputs, model_config, name)
        inputs = tf.keras.layers.Input(observation_space.shape)
        with tf1.variable_scope(tf1.VariableScope(tf1.AUTO_REUSE, 'shared'), reuse=tf1.AUTO_REUSE, auxiliary_name_scope=False):
            last_layer = tf.keras.layers.Dense(units=64, activation=tf.nn.relu, name='fc1')(inputs)
        output = tf.keras.layers.Dense(units=num_outputs, activation=None, name='fc_out')(last_layer)
        vf = tf.keras.layers.Dense(units=1, activation=None, name='value_out')(last_layer)
        self.base_model = tf.keras.models.Model(inputs, [output, vf])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        out, self._value_out = self.base_model(input_dict['obs'])
        return (out, [])

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])