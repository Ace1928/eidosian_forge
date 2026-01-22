from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.core import log
def _ListInstanceEffectiveFirewall(self):
    req = self.compute_message.ComputeInstancesGetEffectiveFirewallsRequest(instance=self.instance.name, networkInterface='nic0', project=self.project.name, zone=self.zone)
    return self.compute_client.instances.GetEffectiveFirewalls(req).firewalls