from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import textwrap
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import exceptions as command_exceptions
from googlecloudsdk.command_lib.app import flags
from googlecloudsdk.command_lib.app import iap_tunnel
from googlecloudsdk.command_lib.app import ssh_common
from googlecloudsdk.command_lib.util.ssh import containers
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _ArgsCommon(parser):
    parser.add_argument('instance', help='The instance ID.')
    parser.add_argument('--container', help='Name of the container within the VM to connect to.')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='Remote command to execute on the VM.')