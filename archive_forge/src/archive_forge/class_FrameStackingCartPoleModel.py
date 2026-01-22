from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.tf_utils import one_hot
from ray.rllib.utils.torch_utils import one_hot as torch_one_hot
class FrameStackingCartPoleModel(TFModelV2):
    """A simple FC model that takes the last n observations as input."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, num_frames=3):
        super(FrameStackingCartPoleModel, self).__init__(obs_space, action_space, None, model_config, name)
        self.num_frames = num_frames
        self.num_outputs = num_outputs
        assert len(obs_space.shape) == 1
        obs = tf.keras.layers.Input(shape=(self.num_frames, obs_space.shape[0]))
        obs_reshaped = tf.keras.layers.Reshape([obs_space.shape[0] * self.num_frames])(obs)
        rewards = tf.keras.layers.Input(shape=self.num_frames)
        rewards_reshaped = tf.keras.layers.Reshape([self.num_frames])(rewards)
        actions = tf.keras.layers.Input(shape=(self.num_frames, self.action_space.n))
        actions_reshaped = tf.keras.layers.Reshape([action_space.n * self.num_frames])(actions)
        input_ = tf.keras.layers.Concatenate(axis=-1)([obs_reshaped, actions_reshaped, rewards_reshaped])
        layer1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(input_)
        layer2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(layer1)
        out = tf.keras.layers.Dense(self.num_outputs)(layer2)
        values = tf.keras.layers.Dense(1)(layer1)
        self.base_model = tf.keras.models.Model([obs, actions, rewards], [out, values])
        self._last_value = None
        self.view_requirements['prev_n_obs'] = ViewRequirement(data_col='obs', shift='-{}:0'.format(num_frames - 1), space=obs_space)
        self.view_requirements['prev_n_rewards'] = ViewRequirement(data_col='rewards', shift='-{}:-1'.format(self.num_frames))
        self.view_requirements['prev_n_actions'] = ViewRequirement(data_col='actions', shift='-{}:-1'.format(self.num_frames), space=self.action_space)

    def forward(self, input_dict, states, seq_lens):
        obs = tf.cast(input_dict['prev_n_obs'], tf.float32)
        rewards = tf.cast(input_dict['prev_n_rewards'], tf.float32)
        actions = one_hot(input_dict['prev_n_actions'], self.action_space)
        out, self._last_value = self.base_model([obs, actions, rewards])
        return (out, [])

    def value_function(self):
        return tf.squeeze(self._last_value, -1)