from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from typing import Mapping, Sequence
from googlecloudsdk.api_lib.run import k8s_object
class EnvVarsAsDictionaryWrapper(k8s_object.ListAsDictionaryWrapper):
    """Wraps a list of env vars in a dict-like object.

  Additionally provides properties to access env vars of specific type in a
  mutable dict-like object.
  """

    def __init__(self, env_vars_to_wrap, env_var_class):
        """Wraps a list of env vars in a dict-like object.

    Args:
      env_vars_to_wrap: list[EnvVar], list of env vars to treat as a dict.
      env_var_class: type of the underlying EnvVar objects.
    """
        super(EnvVarsAsDictionaryWrapper, self).__init__(env_vars_to_wrap)
        self._env_vars = env_vars_to_wrap
        self._env_var_class = env_var_class

    @property
    def literals(self):
        """Mutable dict-like object for env vars with a string literal.

    Note that if neither value nor valueFrom is specified, the list entry will
    be treated as a literal empty string.

    Returns:
      A mutable, dict-like object for managing string literal env vars.
    """
        return k8s_object.KeyValueListAsDictionaryWrapper(self._env_vars, self._env_var_class, filter_func=lambda env_var: env_var.valueFrom is None)

    @property
    def secrets(self):
        """Mutable dict-like object for vars with a secret source type."""

        def _FilterSecretEnvVars(env_var):
            return env_var.valueFrom is not None and env_var.valueFrom.secretKeyRef is not None
        return k8s_object.KeyValueListAsDictionaryWrapper(self._env_vars, self._env_var_class, value_field='valueFrom', filter_func=_FilterSecretEnvVars)

    @property
    def config_maps(self):
        """Mutable dict-like object for vars with a config map source type."""

        def _FilterConfigMapEnvVars(env_var):
            return env_var.valueFrom is not None and env_var.valueFrom.configMapKeyRef is not None
        return k8s_object.KeyValueListAsDictionaryWrapper(self._env_vars, self._env_var_class, value_field='valueFrom', filter_func=_FilterConfigMapEnvVars)