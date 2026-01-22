import collections
from gymnasium.spaces import Space
import logging
import numpy as np
import tree  # pip install dm_tree
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.collectors.sample_collector import SampleCollector
from ray.rllib.evaluation.collectors.agent_collector import AgentCollector
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, concat_samples
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.spaces.space_utils import get_dummy_batch_for_space
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
class _PolicyCollector:
    """Collects already postprocessed (single agent) samples for one policy.

    Samples come in through already postprocessed SampleBatches, which
    contain single episode/trajectory data for a single agent and are then
    appended to this policy's buffers.
    """

    def __init__(self, policy: Policy):
        """Initializes a _PolicyCollector instance.

        Args:
            policy: The policy object.
        """
        self.batches = []
        self.policy = policy
        self.agent_steps = 0

    def add_postprocessed_batch_for_training(self, batch: SampleBatch, view_requirements: ViewRequirementsDict) -> None:
        """Adds a postprocessed SampleBatch (single agent) to our buffers.

        Args:
            batch: An individual agent's (one trajectory)
                SampleBatch to be added to the Policy's buffers.
            view_requirements: The view
                requirements for the policy. This is so we know, whether a
                view-column needs to be copied at all (not needed for
                training).
        """
        self.agent_steps += batch.count
        for view_col, view_req in view_requirements.items():
            if view_col in batch and (not view_req.used_for_training):
                del batch[view_col]
        self.batches.append(batch)

    def build(self):
        """Builds a SampleBatch for this policy from the collected data.

        Also resets all buffers for further sample collection for this policy.

        Returns:
            SampleBatch: The SampleBatch with all thus-far collected data for
                this policy.
        """
        batch = concat_samples(self.batches)
        self.batches = []
        self.agent_steps = 0
        batch.num_grad_updates = self.policy.num_grad_updates
        return batch