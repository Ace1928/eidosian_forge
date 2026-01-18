from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.schedules import PiecewiseSchedule
Defines extra fetches per action computation.

        Args:
            input_dict (Dict[str, TensorType]): The input dict used for the action
                computing forward pass.
            state_batches (List[TensorType]): List of state tensors (empty for
                non-RNNs).
            model (ModelV2): The Model object of the Policy.
            action_dist: The instantiated distribution
                object, resulting from the model's outputs and the given
                distribution class.

        Returns:
            Dict[str, TensorType]: Dict with extra tf fetches to perform per
                action computation.
        