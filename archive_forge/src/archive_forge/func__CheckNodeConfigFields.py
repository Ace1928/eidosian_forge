from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
def _CheckNodeConfigFields(parent_name, parent, spec):
    """Check whether the children of the config option are valid or not.

  Args:
    parent_name: The name of the config option to be checked.
    parent: The config option to be checked.
    spec: The spec defining the expected children and their value type.

  Raises:
    NodeConfigError: if there is any unknown fields or any of the fields doesn't
    satisfy the spec.
  """
    _CheckNodeConfigValueType(parent_name, parent, dict)
    unknown_fields = set(parent.keys()) - set(spec.keys())
    if unknown_fields:
        raise NodeConfigError('unknown fields: {0} in "{1}"'.format(sorted(list(unknown_fields)), parent_name))
    for field_name in parent:
        _CheckNodeConfigValueType(field_name, parent[field_name], spec[field_name])