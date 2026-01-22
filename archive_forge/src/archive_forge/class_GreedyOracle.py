import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import keras_tuner
import numpy as np
from autokeras.engine import tuner as tuner_module
class GreedyOracle(keras_tuner.Oracle):
    """An oracle combining random search and greedy algorithm.

    It groups the HyperParameters into several categories, namely, HyperGraph,
    Preprocessor, Architecture, and Optimization. The oracle tunes each group
    separately using random search. In each trial, it use a greedy strategy to
    generate new values for one of the categories of HyperParameters and use the best
    trial so far for the rest of the HyperParameters values.

    # Arguments
        initial_hps: A list of dictionaries in the form of
            {HyperParameter name (String): HyperParameter value}.
            Each dictionary is one set of HyperParameters, which are used as the
            initial trials for the search. Defaults to None.
        seed: Int. Random seed.
    """

    def __init__(self, initial_hps=None, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.initial_hps = copy.deepcopy(initial_hps) or []
        self._tried_initial_hps = [False] * len(self.initial_hps)

    def get_state(self):
        state = super().get_state()
        state.update({'initial_hps': self.initial_hps, 'tried_initial_hps': self._tried_initial_hps})
        return state

    def set_state(self, state):
        super().set_state(state)
        self.initial_hps = state['initial_hps']
        self._tried_initial_hps = state['tried_initial_hps']

    def _select_hps(self):
        trie = Trie()
        best_hps = self._get_best_hps()
        for hp in best_hps.space:
            if best_hps.is_active(hp) and (not isinstance(hp, keras_tuner.engine.hyperparameters.Fixed)):
                trie.insert(hp.name)
        all_nodes = trie.nodes
        if len(all_nodes) <= 1:
            return []
        probabilities = np.array([1 / node.num_leaves for node in all_nodes])
        sum_p = np.sum(probabilities)
        probabilities = probabilities / sum_p
        node = np.random.choice(all_nodes, p=probabilities)
        return trie.get_hp_names(node)

    def _next_initial_hps(self):
        for index, hps in enumerate(self.initial_hps):
            if not self._tried_initial_hps[index]:
                self._tried_initial_hps[index] = True
                return hps

    def populate_space(self, trial_id):
        if not all(self._tried_initial_hps):
            values = self._next_initial_hps()
            return {'status': keras_tuner.engine.trial.TrialStatus.RUNNING, 'values': values}
        for _ in range(self._max_collisions):
            hp_names = self._select_hps()
            values = self._generate_hp_values(hp_names)
            if values is None:
                continue
            return {'status': keras_tuner.engine.trial.TrialStatus.RUNNING, 'values': values}
        return {'status': keras_tuner.engine.trial.TrialStatus.STOPPED, 'values': None}

    def _get_best_hps(self):
        best_trials = self.get_best_trials()
        if best_trials:
            return best_trials[0].hyperparameters.copy()
        else:
            return self.hyperparameters.copy()

    def _generate_hp_values(self, hp_names):
        best_hps = self._get_best_hps()
        collisions = 0
        while True:
            hps = keras_tuner.HyperParameters()
            for hp in self.hyperparameters.space:
                hps.merge([hp])
                if hps.is_active(hp):
                    if best_hps.is_active(hp.name) and hp.name not in hp_names:
                        hps.values[hp.name] = best_hps.values[hp.name]
                        continue
                    hps.values[hp.name] = hp.random_sample(self._seed_state)
                    self._seed_state += 1
            values = hps.values
            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_so_far:
                collisions += 1
                if collisions <= self._max_collisions:
                    continue
                return None
            self._tried_so_far.add(values_hash)
            break
        return values