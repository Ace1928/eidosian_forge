import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def _get_scoped_config_providers(self, config_property_names):
    scoped_config_providers = []
    if not isinstance(config_property_names, list):
        config_property_names = [config_property_names]
    for config_property_name in config_property_names:
        scoped_config_providers.append(ScopedConfigProvider(config_var_name=config_property_name, session=self._session))
    return scoped_config_providers