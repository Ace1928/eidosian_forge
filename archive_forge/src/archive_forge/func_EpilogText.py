from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import networks_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.networks import flags
from googlecloudsdk.command_lib.compute.networks import network_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_projector
def EpilogText(network_name):
    """Text for firewall warning."""
    message = '\n      Instances on this network will not be reachable until firewall rules\n      are created. As an example, you can allow all internal traffic between\n      instances as well as SSH, RDP, and ICMP by running:\n\n      $ gcloud compute firewall-rules create <FIREWALL_NAME> --network {0} --allow tcp,udp,icmp --source-ranges <IP_RANGE>\n      $ gcloud compute firewall-rules create <FIREWALL_NAME> --network {0} --allow tcp:22,tcp:3389,icmp\n      '.format(network_name)
    log.status.Print(textwrap.dedent(message))