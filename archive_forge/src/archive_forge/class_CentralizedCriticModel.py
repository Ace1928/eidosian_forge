from gymnasium.spaces import Box
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
class CentralizedCriticModel(TFModelV2):
    """Multi-agent model that implements a centralized value function."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CentralizedCriticModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space, num_outputs, model_config, name)
        obs = tf.keras.layers.Input(shape=(6,), name='obs')
        opp_obs = tf.keras.layers.Input(shape=(6,), name='opp_obs')
        opp_act = tf.keras.layers.Input(shape=(2,), name='opp_act')
        concat_obs = tf.keras.layers.Concatenate(axis=1)([obs, opp_obs, opp_act])
        central_vf_dense = tf.keras.layers.Dense(16, activation=tf.nn.tanh, name='c_vf_dense')(concat_obs)
        central_vf_out = tf.keras.layers.Dense(1, activation=None, name='c_vf_out')(central_vf_dense)
        self.central_vf = tf.keras.Model(inputs=[obs, opp_obs, opp_act], outputs=central_vf_out)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def central_value_function(self, obs, opponent_obs, opponent_actions):
        return tf.reshape(self.central_vf([obs, opponent_obs, tf.one_hot(tf.cast(opponent_actions, tf.int32), 2)]), [-1])

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()