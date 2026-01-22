import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import keras_tuner
import numpy as np
from autokeras.engine import tuner as tuner_module
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