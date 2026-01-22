import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def _resolve_default_values_by_mode(self, mode):
    default_config = self._base_default_config.copy()
    modifications = self._modes.get(mode)
    for config_var in modifications:
        default_value = default_config[config_var]
        modification_dict = modifications[config_var]
        modification = list(modification_dict.keys())[0]
        modification_value = modification_dict[modification]
        if modification == 'multiply':
            default_value *= modification_value
        elif modification == 'add':
            default_value += modification_value
        elif modification == 'override':
            default_value = modification_value
        default_config[config_var] = default_value
    return default_config