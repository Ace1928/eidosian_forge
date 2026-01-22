import argparse
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.visionnet import VisionNetwork as MyVisionNetwork
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO, LEARNER_STATS_KEY
from ray.tune.registry import get_trainable_cls
class MyKerasQModel(DistributionalQTFModel):
    """Custom model for DQN."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super(MyKerasQModel, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name='observations')
        layer_1 = tf.keras.layers.Dense(128, name='my_layer1', activation=tf.nn.relu, kernel_initializer=normc_initializer(1.0))(self.inputs)
        layer_out = tf.keras.layers.Dense(num_outputs, name='my_out', activation=tf.nn.relu, kernel_initializer=normc_initializer(1.0))(layer_1)
        self.base_model = tf.keras.Model(self.inputs, layer_out)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict['obs'])
        return (model_out, state)

    def metrics(self):
        return {'foo': tf.constant(42.0)}