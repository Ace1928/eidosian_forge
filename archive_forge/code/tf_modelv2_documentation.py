import contextlib
import gymnasium as gym
import re
from typing import Dict, List, Union
from ray.util import log_once
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict, TensorType
Register the given list of variables with this model.