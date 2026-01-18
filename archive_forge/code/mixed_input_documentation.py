from types import FunctionType
from typing import Dict
import numpy as np
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.typing import SampleBatchType
from ray.tune.registry import registry_get_input, registry_contains_input
Initialize a MixedInput.

        Args:
            dist: dict mapping JSONReader paths or "sampler" to
                probabilities. The probabilities must sum to 1.0.
            ioctx: current IO context object.
        