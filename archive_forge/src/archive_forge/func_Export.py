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
def Export(self, args, resource_uri):
    """Exports a single resource's configuration file."""
    normalized_resource_uri = _NormalizeUri(resource_uri)
    with progress_tracker.ProgressTracker(message='Exporting resources', aborted_message='Aborted Export.'):
        cmd = self._GetBinaryExportCommand(args=args, command_name='export', resource_uri=normalized_resource_uri)
        exit_code, output_value, error_value = _ExecuteBinary(cmd)
    if exit_code != 0:
        if 'resource not found' in error_value:
            raise ResourceNotFoundException('Could not fetch resource: \n - The resource [{}] does not exist.'.format(normalized_resource_uri))
        elif 'Error 403' in error_value:
            raise ClientException('Permission Denied during export. Please ensure resource API is enabled for resource [{}] and Cloud IAM permissions are set properly.'.format(resource_uri))
        else:
            raise ClientException('Error executing export:: [{}]'.format(error_value))
    if not self._OutputToFileOrDir(args.path):
        log.out.Print(output_value)
    log.status.Print('Exported successfully.')
    return exit_code