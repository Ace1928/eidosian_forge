import abc
from typing import Callable
import dataclasses
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
@kullback_leibler.RegisterKL(TanhTransformedDistribution, TanhTransformedDistribution)
def _kl_transformed(a, b, name='kl_transformed'):
    return kullback_leibler.kl_divergence(a.distribution, b.distribution, name=name)