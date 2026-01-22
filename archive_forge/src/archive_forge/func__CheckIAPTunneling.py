from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.core import log
def _CheckIAPTunneling(self):
    firewall_list = self._ListInstanceEffectiveFirewall()
    for firewall in firewall_list:
        if self._HasValidateIAPTunnelingRule(firewall):
            return
    self.issues['iap'] = IAP_MESSAGE