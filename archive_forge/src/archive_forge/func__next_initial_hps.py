import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import keras_tuner
import numpy as np
from autokeras.engine import tuner as tuner_module
def _next_initial_hps(self):
    for index, hps in enumerate(self.initial_hps):
        if not self._tried_initial_hps[index]:
            self._tried_initial_hps[index] = True
            return hps