import gymnasium as gym
import numpy as np
import os
import ray._private.utils
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
Simple example of writing experiences to a file using JsonWriter.