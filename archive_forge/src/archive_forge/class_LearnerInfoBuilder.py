from collections import defaultdict
import numpy as np
import tree  # pip install dm_tree
from typing import Dict
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.typing import PolicyID
@DeveloperAPI
class LearnerInfoBuilder:

    def __init__(self, num_devices: int=1):
        self.num_devices = num_devices
        self.results_all_towers = defaultdict(list)
        self.is_finalized = False

    def add_learn_on_batch_results(self, results: Dict, policy_id: PolicyID=DEFAULT_POLICY_ID) -> None:
        """Adds a policy.learn_on_(loaded)?_batch() result to this builder.

        Args:
            results: The results returned by Policy.learn_on_batch or
                Policy.learn_on_loaded_batch.
            policy_id: The policy's ID, whose learn_on_(loaded)_batch method
                returned `results`.
        """
        assert not self.is_finalized, 'LearnerInfo already finalized! Cannot add more results.'
        if 'tower_0' not in results:
            self.results_all_towers[policy_id].append(results)
        else:
            self.results_all_towers[policy_id].append(tree.map_structure_with_path(lambda p, *s: _all_tower_reduce(p, *s), *(results.pop('tower_{}'.format(tower_num)) for tower_num in range(self.num_devices))))
            for k, v in results.items():
                if k == LEARNER_STATS_KEY:
                    for k1, v1 in results[k].items():
                        self.results_all_towers[policy_id][-1][LEARNER_STATS_KEY][k1] = v1
                else:
                    self.results_all_towers[policy_id][-1][k] = v

    def add_learn_on_batch_results_multi_agent(self, all_policies_results: Dict) -> None:
        """Adds multiple policy.learn_on_(loaded)?_batch() results to this builder.

        Args:
            all_policies_results: The results returned by all Policy.learn_on_batch or
                Policy.learn_on_loaded_batch wrapped as a dict mapping policy ID to
                results.
        """
        for pid, result in all_policies_results.items():
            if pid != 'batch_count':
                self.add_learn_on_batch_results(result, policy_id=pid)

    def finalize(self):
        self.is_finalized = True
        info = {}
        for policy_id, results_all_towers in self.results_all_towers.items():
            info[policy_id] = tree.map_structure_with_path(_all_tower_reduce, *results_all_towers)
        return info