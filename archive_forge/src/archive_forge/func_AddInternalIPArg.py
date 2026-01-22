from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
import sys
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import iap_tunnel
from googlecloudsdk.command_lib.compute import network_troubleshooter
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute import user_permission_troubleshooter
from googlecloudsdk.command_lib.compute import vm_boot_troubleshooter
from googlecloudsdk.command_lib.compute import vm_status_troubleshooter
from googlecloudsdk.command_lib.compute import vpc_troubleshooter
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.util.ssh import containers
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
def AddInternalIPArg(group):
    group.add_argument('--internal-ip', default=False, action='store_true', help='        Connect to instances using their internal IP addresses rather than their\n        external IP addresses. Use this to connect from one instance to another\n        on the same VPC network, over a VPN connection, or between two peered\n        VPC networks.\n\n        For this connection to work, you must configure your networks and\n        firewall to allow SSH connections to the internal IP address of\n        the instance to which you want to connect.\n\n        To learn how to use this flag, see\n        [](https://cloud.google.com/compute/docs/instances/connecting-advanced#sshbetweeninstances).\n        ')