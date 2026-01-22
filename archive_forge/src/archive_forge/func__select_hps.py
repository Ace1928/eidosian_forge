import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import keras_tuner
import numpy as np
from autokeras.engine import tuner as tuner_module
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