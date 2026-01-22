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
class GumbelSoftmax(TFActionDistribution):
    """GumbelSoftmax distr. (for differentiable sampling in discr. actions

    The Gumbel Softmax distribution [1] (also known as the Concrete [2]
    distribution) is a close cousin of the relaxed one-hot categorical
    distribution, whose tfp implementation we will use here plus
    adjusted `sample_...` and `log_prob` methods. See discussion at [0].

    [0] https://stackoverflow.com/questions/56226133/
    soft-actor-critic-with-discrete-action-space

    [1] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017):
    https://arxiv.org/abs/1611.01144
    [2] The Concrete Distribution: A Continuous Relaxation of Discrete Random
    Variables (Maddison et al, 2017) https://arxiv.org/abs/1611.00712
    """

    def __init__(self, inputs: List[TensorType], model: ModelV2=None, temperature: float=1.0):
        """Initializes a GumbelSoftmax distribution.

        Args:
            temperature: Temperature parameter. For low temperatures,
                the expected value approaches a categorical random variable.
                For high temperatures, the expected value approaches a uniform
                distribution.
        """
        assert temperature >= 0.0
        self.dist = tfp.distributions.RelaxedOneHotCategorical(temperature=temperature, logits=inputs)
        self.probs = tf.nn.softmax(self.dist._distribution.logits)
        super().__init__(inputs, model)

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        return self.probs

    @override(ActionDistribution)
    def logp(self, x: TensorType) -> TensorType:
        if x.shape != self.dist.logits.shape:
            values = tf.one_hot(x, self.dist.logits.shape.as_list()[-1], dtype=tf.float32)
            assert values.shape == self.dist.logits.shape, (values.shape, self.dist.logits.shape)
        return -tf.reduce_sum(-x * tf.nn.log_softmax(self.dist.logits, axis=-1), axis=-1)

    @override(TFActionDistribution)
    def _build_sample_op(self) -> TensorType:
        return self.dist.sample()

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space: gym.Space, model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        return action_space.n