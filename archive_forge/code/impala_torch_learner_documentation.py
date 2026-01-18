from typing import Mapping
from ray.rllib.algorithms.impala.impala_learner import (
from ray.rllib.algorithms.impala.torch.vtrace_torch_v2 import (
from ray.rllib.core.rl_module.rl_module import ModuleID
from ray.rllib.core.learner.learner import ENTROPY_KEY
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import TensorType
Implements the IMPALA loss function in torch.