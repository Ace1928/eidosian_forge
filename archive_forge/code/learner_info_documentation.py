from collections import defaultdict
import numpy as np
import tree  # pip install dm_tree
from typing import Dict
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.typing import PolicyID
Adds multiple policy.learn_on_(loaded)?_batch() results to this builder.

        Args:
            all_policies_results: The results returned by all Policy.learn_on_batch or
                Policy.learn_on_loaded_batch wrapped as a dict mapping policy ID to
                results.
        