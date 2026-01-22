import functools
import gymnasium as gym
from math import log
import numpy as np
import tree  # pip install dm_tree
from typing import Optional
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils import MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT, SMALL_NUMBER
from ray.rllib.utils.annotations import override, DeveloperAPI, ExperimentalAPI
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import TensorType, List, Union, Tuple, ModelConfigDict
@DeveloperAPI
class DiagGaussian(TFActionDistribution):
    """Action distribution where each vector element is a gaussian.

    The first half of the input vector defines the gaussian means, and the
    second half the gaussian standard deviations.
    """

    def __init__(self, inputs: List[TensorType], model: ModelV2, *, action_space: Optional[gym.spaces.Space]=None):
        mean, log_std = tf.split(inputs, 2, axis=1)
        self.mean = mean
        self.log_std = log_std
        self.std = tf.exp(log_std)
        self.zero_action_dim = action_space and action_space.shape == ()
        super().__init__(inputs, model)

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        return self.mean

    @override(ActionDistribution)
    def logp(self, x: TensorType) -> TensorType:
        if int(tf.shape(x).shape[0]) == 1:
            x = tf.expand_dims(x, axis=1)
        return -0.5 * tf.reduce_sum(tf.math.square((tf.cast(x, tf.float32) - self.mean) / self.std), axis=1) - 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[1], tf.float32) - tf.reduce_sum(self.log_std, axis=1)

    @override(ActionDistribution)
    def kl(self, other: ActionDistribution) -> TensorType:
        assert isinstance(other, DiagGaussian)
        return tf.reduce_sum(other.log_std - self.log_std + (tf.math.square(self.std) + tf.math.square(self.mean - other.mean)) / (2.0 * tf.math.square(other.std)) - 0.5, axis=1)

    @override(ActionDistribution)
    def entropy(self) -> TensorType:
        return tf.reduce_sum(self.log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=1)

    @override(TFActionDistribution)
    def _build_sample_op(self) -> TensorType:
        sample = self.mean + self.std * tf.random.normal(tf.shape(self.mean))
        if self.zero_action_dim:
            return tf.squeeze(sample, axis=-1)
        return sample

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space: gym.Space, model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        return np.prod(action_space.shape, dtype=np.int32) * 2