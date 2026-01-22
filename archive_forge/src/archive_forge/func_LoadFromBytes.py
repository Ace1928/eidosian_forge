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
@classmethod
def LoadFromBytes(cls, raw_data: bytes, path: str=None) -> Kubeconfig:
    """Parse a YAML kubeconfig.

    Args:
      raw_data: The YAML data to parse
      path: The path to associate with the data. Defaults to calling
        `Kubeconfig.DefaultPath()`.

    Returns:
      A `Kubeconfig` instance.

    Raises:
      Error: The data is not valid YAML.
    """
    try:
        data = yaml.load(raw_data)
    except yaml.Error as error:
        raise Error(f'unable to parse kubeconfig bytes: {error.inner_error}')
    cls._Validate(data)
    if not path:
        path = cls.DefaultPath()
    return cls(data, path)