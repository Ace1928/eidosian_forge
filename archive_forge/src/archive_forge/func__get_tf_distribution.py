import gymnasium as gym
import tree
import numpy as np
from typing import Optional, List, Mapping, Iterable, Dict
import abc
from ray.rllib.models.distributions import Distribution
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.typing import TensorType, Union, Tuple
@override(TfDistribution)
def _get_tf_distribution(self, loc, scale) -> 'tfp.distributions.Distribution':
    return tfp.distributions.Normal(loc=loc, scale=scale)