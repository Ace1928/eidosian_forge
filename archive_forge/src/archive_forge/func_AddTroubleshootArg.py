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
def AddTroubleshootArg(parser):
    parser.add_argument('--troubleshoot', action='store_true', help="          If you can't connect to a virtual machine (VM) instance using SSH, you can investigate the problem using the `--troubleshoot` flag:\n\n            $ {command} VM_NAME --zone=ZONE --troubleshoot [--tunnel-through-iap]\n\n          The troubleshoot flag runs tests and returns recommendations for four types of issues:\n          - VM status\n          - Network connectivity\n          - User permissions\n          - Virtual Private Cloud (VPC) settings\n          - VM boot\n\n          If you specify the `--tunnel-through-iap` flag, the tool also checks IAP port forwarding.\n          ")