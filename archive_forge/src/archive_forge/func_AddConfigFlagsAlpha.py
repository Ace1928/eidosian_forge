from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
import six
def AddConfigFlagsAlpha(worker_pools):
    """Add config flags."""
    worker_pools.add_argument('--memory', type=arg_parsers.BinarySize(default_unit='GB'), hidden=True, help='Machine memory required to run a build.')
    worker_pools.add_argument('--vcpu-count', type=float, hidden=True, help='Machine vCPU count required to run a build.')