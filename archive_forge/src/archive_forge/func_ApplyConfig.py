from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import errno
import io
import os
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.command_lib.asset import utils as asset_utils
from googlecloudsdk.command_lib.util.resource_map.declarative import resource_name_translator
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.util import files
import six
def ApplyConfig(self, input_path, try_resolve_refs=False):
    """Call apply from config-connector binary.

    Applys the KRM config file specified by `path`, creating or updating the
    related GCP resources. If `try_resolve_refs` is supplied, then command will
    attempt to resolve the references to related resources in config,
    creating a directed graph of related resources and apply them in order.

    Args:
      input_path: string, KRM config file to apply.
      try_resolve_refs: boolean, if true attempt to resolve the references to
      related resources in config.

    Returns:
      Yaml Object representing the updated state of the resource if successful.

    Raises:
      ApplyException: if an error occurs applying config.
      ApplyPathException: if an error occurs reading file path.
    """
    del try_resolve_refs
    if not input_path or not input_path.strip() or (not os.path.isfile(input_path)):
        raise ApplyPathException('Resource file path [{}] not found.'.format(input_path))
    cmd = [self._export_service, 'apply', '-i', input_path, '--oauth2-token', self._GetToken()]
    exit_code, output_value, error_value = _ExecuteBinary(cmd)
    if exit_code != 0:
        raise ApplyException('Error occured while applying configuration path [{}]: [{}]'.format(input_path, error_value))
    return yaml.load(output_value)