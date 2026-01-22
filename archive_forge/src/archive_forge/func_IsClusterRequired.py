from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import container_command_util
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from six.moves import input  # pylint: disable=redefined-builtin
def IsClusterRequired(self, args):
    """Returns if failure getting the cluster should be an error."""
    return bool(getattr(args, 'maintenance_window_end', False) or getattr(args, 'clear_maintenance_window', False) or getattr(args, 'add_maintenance_exclusion_end', False) or getattr(args, 'remove_maintenance_exclusion', False) or getattr(args, 'add_cross_connect_subnetworks', False) or getattr(args, 'remove_cross_connect_subnetworks', False) or getattr(args, 'clear_cross_connect_subnetworks', False) or getattr(args, 'enable_google_cloud_access', False))