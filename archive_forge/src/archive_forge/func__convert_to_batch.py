import logging
import gymnasium as gym
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import SampleBatchType
from typing import Dict
def _convert_to_batch(dataset: Dict) -> SampleBatchType:
    d = {}
    d[SampleBatch.OBS] = dataset['observations']
    d[SampleBatch.ACTIONS] = dataset['actions']
    d[SampleBatch.NEXT_OBS] = dataset['next_observations']
    d[SampleBatch.REWARDS] = dataset['rewards']
    d[SampleBatch.TERMINATEDS] = dataset['terminals']
    return SampleBatch(d)