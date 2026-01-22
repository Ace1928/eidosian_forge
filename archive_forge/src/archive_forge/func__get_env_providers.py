import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def _get_env_providers(self, env_var_names):
    env_var_providers = []
    if not isinstance(env_var_names, list):
        env_var_names = [env_var_names]
    for env_var_name in env_var_names:
        env_var_providers.append(EnvironmentProvider(name=env_var_name, env=self._environ))
    return env_var_providers