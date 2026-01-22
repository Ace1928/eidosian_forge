from collections import Counter
import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from gymnasium.spaces import Dict as GymDict
from gymnasium.spaces import Tuple as GymTuple
import inspect
import logging
import numpy as np
import os
import pprint
import random
import re
import time
import tree  # pip install dm_tree
from typing import (
import yaml
import ray
from ray import air, tune
from ray.rllib.env.wrappers.atari_wrappers import is_atari, wrap_deepmind
from ray.rllib.utils.framework import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.tune import CLIReporter, run_experiments
class ModelChecker:
    """Helper class to compare architecturally identical Models across frameworks.

    Holds a ModelConfig, such that individual models can be added simply via their
    framework string (by building them with config.build(framework=...).
    A call to `check()` forces all added models to be compared in terms of their
    number of trainable and non-trainable parameters, as well as, their
    computation results given a common weights structure and values and identical
    inputs to the models.
    """

    def __init__(self, config):
        self.config = config
        self.param_counts = {}
        self.output_values = {}
        self.random_fill_input_value = np.random.uniform(-0.01, 0.01)
        self.models = {}

    def add(self, framework: str='torch') -> Any:
        """Builds a new Model for the given framework."""
        model = self.models[framework] = self.config.build(framework=framework)
        from ray.rllib.core.models.specs.specs_dict import SpecDict
        if isinstance(model.input_specs, SpecDict):
            inputs = {}
            for key, spec in model.input_specs.items():
                dict_ = inputs
                for i, sub_key in enumerate(key):
                    if sub_key not in dict_:
                        dict_[sub_key] = {}
                    if i < len(key) - 1:
                        dict_ = dict_[sub_key]
                if spec is not None:
                    dict_[sub_key] = spec.fill(self.random_fill_input_value)
                else:
                    dict_[sub_key] = None
        else:
            inputs = model.input_specs.fill(self.random_fill_input_value)
        outputs = model(inputs)
        model._set_to_dummy_weights(value_sequence=(self.random_fill_input_value,))
        comparable_outputs = model(inputs)
        self.param_counts[framework] = model.get_num_parameters()
        if framework == 'torch':
            self.output_values[framework] = tree.map_structure(lambda s: s.detach().numpy() if s is not None else None, comparable_outputs)
        else:
            self.output_values[framework] = tree.map_structure(lambda s: s.numpy() if s is not None else None, comparable_outputs)
        return outputs

    def check(self):
        """Compares all added Models with each other and possibly raises errors."""
        main_key = next(iter(self.models.keys()))
        for c in self.param_counts.values():
            check(c, self.param_counts[main_key])
        for v in self.output_values.values():
            check(v, self.output_values[main_key], atol=0.0005)