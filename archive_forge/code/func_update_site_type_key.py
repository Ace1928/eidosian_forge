from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
def update_site_type_key(self, config):
    """
        Replace 'site_type' key with 'type' in the config.

        Parameters:
            config (list or dict) - Configuration details.

        Returns:
            updated_config (list or dict) - Updated config after replacing the keys.
        """
    if isinstance(config, dict):
        new_config = {}
        for key, value in config.items():
            if key == 'site_type':
                new_key = 'type'
            else:
                new_key = re.sub('([a-z0-9])([A-Z])', '\\1_\\2', key).lower()
            new_value = self.update_site_type_key(value)
            new_config[new_key] = new_value
    elif isinstance(config, list):
        return [self.update_site_type_key(item) for item in config]
    else:
        return config
    return new_config