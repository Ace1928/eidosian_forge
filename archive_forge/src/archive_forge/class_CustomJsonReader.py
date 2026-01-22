import argparse
import os
import ray
from ray import air, tune
from ray.rllib.offline import JsonReader, ShuffledInput, IOContext, InputReader
from ray.tune.registry import get_trainable_cls, register_input
class CustomJsonReader(JsonReader):
    """
    Example custom InputReader implementation (extended from JsonReader).

    This gets wrapped in ShuffledInput to comply with offline rl algorithms.
    """

    def __init__(self, ioctx: IOContext):
        """
        The constructor must take an IOContext to be used in the input config.
        Args:
            ioctx: use this to access the `input_config` arguments.
        """
        super().__init__(ioctx.input_config['input_files'], ioctx)