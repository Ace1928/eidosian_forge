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
class MultiCategorical(TFActionDistribution):
    """MultiCategorical distribution for MultiDiscrete action spaces."""

    def __init__(self, inputs: List[TensorType], model: ModelV2, input_lens: Union[List[int], np.ndarray, Tuple[int, ...]], action_space=None):
        ActionDistribution.__init__(self, inputs, model)
        self.cats = [Categorical(input_, model) for input_ in tf.split(inputs, input_lens, axis=1)]
        self.action_space = action_space
        if self.action_space is None:
            self.action_space = gym.spaces.MultiDiscrete([c.inputs.shape[1] for c in self.cats])
        self.sample_op = self._build_sample_op()
        self.sampled_action_logp_op = self.logp(self.sample_op)

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        sample_ = tf.stack([cat.deterministic_sample() for cat in self.cats], axis=1)
        if isinstance(self.action_space, gym.spaces.Box):
            return tf.cast(tf.reshape(sample_, [-1] + list(self.action_space.shape)), self.action_space.dtype)
        return sample_

    @override(ActionDistribution)
    def logp(self, actions: TensorType) -> TensorType:
        if isinstance(actions, tf.Tensor):
            if isinstance(self.action_space, gym.spaces.Box):
                actions = tf.reshape(actions, [-1, int(np.prod(self.action_space.shape))])
            elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
                actions.set_shape((None, len(self.cats)))
            actions = tf.unstack(tf.cast(actions, tf.int32), axis=1)
        logps = tf.stack([cat.logp(act) for cat, act in zip(self.cats, actions)])
        return tf.reduce_sum(logps, axis=0)

    @override(ActionDistribution)
    def multi_entropy(self) -> TensorType:
        return tf.stack([cat.entropy() for cat in self.cats], axis=1)

    @override(ActionDistribution)
    def entropy(self) -> TensorType:
        return tf.reduce_sum(self.multi_entropy(), axis=1)

    @override(ActionDistribution)
    def multi_kl(self, other: ActionDistribution) -> TensorType:
        return tf.stack([cat.kl(oth_cat) for cat, oth_cat in zip(self.cats, other.cats)], axis=1)

    @override(ActionDistribution)
    def kl(self, other: ActionDistribution) -> TensorType:
        return tf.reduce_sum(self.multi_kl(other), axis=1)

    @override(TFActionDistribution)
    def _build_sample_op(self) -> TensorType:
        sample_op = tf.stack([cat.sample() for cat in self.cats], axis=1)
        if isinstance(self.action_space, gym.spaces.Box):
            return tf.cast(tf.reshape(sample_op, [-1] + list(self.action_space.shape)), dtype=self.action_space.dtype)
        return sample_op

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space: gym.Space, model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        if isinstance(action_space, gym.spaces.Box):
            assert action_space.dtype.name.startswith('int')
            low_ = np.min(action_space.low)
            high_ = np.max(action_space.high)
            assert np.all(action_space.low == low_)
            assert np.all(action_space.high == high_)
            return np.prod(action_space.shape, dtype=np.int32) * (high_ - low_ + 1)
        else:
            return np.sum(action_space.nvec)