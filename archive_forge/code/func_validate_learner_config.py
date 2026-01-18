import collections
import pickle
import threading
import time
import timeit
from absl import flags
from absl import logging
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.distribute import values as values_lib  
from tensorflow.python.framework import composite_tensor  
from tensorflow.python.framework import tensor_conversion_registry  
def validate_learner_config(config, num_hosts=1):
    """Shared part of learner config validation."""
    assert config.num_envs > 0
    assert config.env_batch_size > 0
    if config.inference_batch_size == -1:
        config.inference_batch_size = max(config.env_batch_size, config.num_envs // (2 * num_hosts))
    assert config.inference_batch_size > 0
    assert config.inference_batch_size % config.env_batch_size == 0, 'Learner-side batch size (=%d) must be exact multiple of the actor-side batch size (=%d).' % (config.inference_batch_size, config.env_batch_size)
    assert config.num_envs >= config.inference_batch_size * num_hosts, 'Inference batch size is bigger than the number of environments.'