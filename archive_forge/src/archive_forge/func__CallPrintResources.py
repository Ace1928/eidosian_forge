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
def _CallPrintResources(self, output_format='table'):
    """Calls `print-resources` on the underlying binary."""
    cmd = [self._export_service, 'print-resources', '--output-format', output_format]
    exit_code, output_value, error_value = _ExecuteBinary(cmd)
    if exit_code != 0:
        raise ClientException('Error occured while listing available resources: [{}]'.format(error_value))
    return output_value