import abc
from typing import Callable
import dataclasses
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
class TanhTransformedDistribution(tfd.TransformedDistribution):
    """Distribution followed by tanh."""

    def __init__(self, distribution, threshold=0.999, validate_args=False):
        """Initialize the distribution.

    Args:
      distribution: The distribution to transform.
      threshold: Clipping value of the action when computing the logprob.
      validate_args: Passed to super class.
    """
        super().__init__(distribution=distribution, bijector=tfp.bijectors.Tanh(), validate_args=validate_args)
        self._threshold = threshold
        inverse_threshold = self.bijector.inverse(threshold)
        log_epsilon = tf.math.log(1.0 - threshold)
        self._log_prob_left = self.distribution.log_cdf(-inverse_threshold) - log_epsilon
        self._log_prob_right = self.distribution.log_survival_function(inverse_threshold) - log_epsilon

    def log_prob(self, event):
        event = tf.clip_by_value(event, -self._threshold, self._threshold)
        return tf.where(event <= -self._threshold, self._log_prob_left, tf.where(event >= self._threshold, self._log_prob_right, super().log_prob(event)))

    def mode(self):
        return self.bijector.forward(self.distribution.mode())

    def mean(self):
        return self.bijector.forward(self.distribution.mean())

    def entropy(self, seed=None):
        return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(self.distribution.sample(seed=seed), event_ndims=0)