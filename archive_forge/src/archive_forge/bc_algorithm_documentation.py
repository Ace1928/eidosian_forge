from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule
from ray.rllib.core.testing.torch.bc_learner import BCTorchLearner
from ray.rllib.core.testing.tf.bc_module import DiscreteBCTFModule
from ray.rllib.core.testing.tf.bc_learner import BCTfLearner
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ResultDict
Contains example implementation of a custom algorithm.

Note: It doesn't include any real use-case functionality; it only serves as an example
to test the algorithm construction and customization.
