import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
class DefaultConfigResolver:

    def __init__(self, default_config_data):
        self._base_default_config = default_config_data['base']
        self._modes = default_config_data['modes']
        self._resolved_default_configurations = {}

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

    def get_default_modes(self):
        default_modes = ['legacy', 'auto']
        default_modes.extend(self._modes.keys())
        return default_modes

    def get_default_config_values(self, mode):
        if mode not in self._resolved_default_configurations:
            defaults = self._resolve_default_values_by_mode(mode)
            self._resolved_default_configurations[mode] = defaults
        return self._resolved_default_configurations[mode]