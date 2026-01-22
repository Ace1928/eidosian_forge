import contextlib
import gymnasium as gym
import re
from typing import Dict, List, Union
from ray.util import log_once
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict, TensorType
@staticmethod
def _find_sub_modules(current_key, struct):
    if isinstance(struct, tf.keras.models.Model) or isinstance(struct, tf.Module):
        ret = {}
        for var in struct.variables:
            name = re.sub('/', '.', var.name)
            key = current_key + '.' + name
            ret[key] = var
        return ret
    elif isinstance(struct, TFModelV2):
        return {current_key + '.' + key: var for key, var in struct.variables(as_dict=True).items()}
    elif isinstance(struct, tf.Variable):
        return {current_key: struct}
    elif isinstance(struct, (tuple, list)):
        ret = {}
        for i, value in enumerate(struct):
            sub_vars = TFModelV2._find_sub_modules(current_key + '_{}'.format(i), value)
            ret.update(sub_vars)
        return ret
    elif isinstance(struct, dict):
        if current_key:
            current_key += '_'
        ret = {}
        for key, value in struct.items():
            sub_vars = TFModelV2._find_sub_modules(current_key + str(key), value)
            ret.update(sub_vars)
        return ret
    return {}