from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import unicode_literals
import os
from typing import Any
from googlecloudsdk.api_lib.container import kubeconfig as container_kubeconfig
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
def User(name, auth_provider=None):
    """Generate and return a user kubeconfig object.

  Args:
    name: str, nickname for this user entry.
    auth_provider: str, authentication provider if not using `exec`. `exec` may
      still be used regardless of this parameter's value.
  Returns:
    dict, valid kubeconfig user entry.

  Raises:
    Error: if no auth_provider is not provided when `exec` is not used.
  """
    return container_kubeconfig.User(name=name, auth_provider=auth_provider)