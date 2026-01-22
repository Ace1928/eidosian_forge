import copy
import os
import traceback
import warnings
from keras_tuner.src import backend
from keras_tuner.src import config as config_module
from keras_tuner.src import errors
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.distribute import utils as dist_utils
from keras_tuner.src.engine import hypermodel as hm_module
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import stateful
from keras_tuner.src.engine import trial as trial_module
from keras_tuner.src.engine import tuner_utils
def _activate_all_conditions(self):
    scopes_never_active = []
    scopes_once_active = []
    hp = self.oracle.get_space()
    while True:
        self.hypermodel.build(hp)
        self.oracle.update_space(hp)
        for conditions in hp.active_scopes:
            if conditions not in scopes_once_active:
                scopes_once_active.append(copy.deepcopy(conditions))
            if conditions in scopes_never_active:
                scopes_never_active.remove(conditions)
        for conditions in hp.inactive_scopes:
            if conditions not in scopes_once_active:
                scopes_never_active.append(copy.deepcopy(conditions))
        if not scopes_never_active:
            break
        hp = self.oracle.get_space()
        conditions = scopes_never_active[0]
        for condition in conditions:
            hp.values[condition.name] = condition.values[0]
        hp.ensure_active_values()