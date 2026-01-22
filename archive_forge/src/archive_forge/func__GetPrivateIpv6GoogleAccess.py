from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.dataproc import compute_helpers
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from stdin.
def _GetPrivateIpv6GoogleAccess(dataproc, private_ipv6_google_access_type):
    """Get PrivateIpv6GoogleAccess enum value.

  Converts private_ipv6_google_access_type argument value to
  PrivateIpv6GoogleAccess API enum value.

  Args:
    dataproc: Dataproc API definition
    private_ipv6_google_access_type: argument value

  Returns:
    PrivateIpv6GoogleAccess API enum value
  """
    if private_ipv6_google_access_type == 'inherit-subnetwork':
        return dataproc.messages.GceClusterConfig.PrivateIpv6GoogleAccessValueValuesEnum('INHERIT_FROM_SUBNETWORK')
    if private_ipv6_google_access_type == 'outbound':
        return dataproc.messages.GceClusterConfig.PrivateIpv6GoogleAccessValueValuesEnum('OUTBOUND')
    if private_ipv6_google_access_type == 'bidirectional':
        return dataproc.messages.GceClusterConfig.PrivateIpv6GoogleAccessValueValuesEnum('BIDIRECTIONAL')
    if private_ipv6_google_access_type is None:
        return None
    raise exceptions.ArgumentError('Unsupported --private-ipv6-google-access-type flag value: ' + private_ipv6_google_access_type)