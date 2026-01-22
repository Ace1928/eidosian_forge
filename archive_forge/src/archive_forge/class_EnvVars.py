from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.kuberun import kubernetesobject
from googlecloudsdk.api_lib.kuberun import structuredout
class EnvVars:
    """Represents the list of env vars/secrets/config maps.

  Provides properties to access the various type of env vars.
  """

    def __init__(self, env_var_list):
        if env_var_list:
            self._env_var_list = env_var_list
        else:
            self._env_var_list = dict()

    @property
    def literals(self):
        return {env['name']: env.get('value') for env in self._env_var_list if env.get('valueFrom') is None}

    @property
    def secrets(self):
        return {env['name']: EnvValueFrom(env.get('valueFrom')) for env in self._env_var_list if env.get('valueFrom') and env.get('valueFrom').get('secretKeyRef')}

    @property
    def config_maps(self):
        return {env['name']: EnvValueFrom(env.get('valueFrom')) for env in self._env_var_list if env.get('valueFrom') and env.get('valueFrom').get('configMapKeyRef')}