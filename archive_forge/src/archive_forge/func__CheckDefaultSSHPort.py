from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.core import log
def _CheckDefaultSSHPort(self):
    firewall_list = self._ListInstanceEffectiveFirewall()
    for firewall in firewall_list:
        if self._HasSSHProtocalAndPort(firewall):
            return
    self.issues['default_ssh_port'] = DEFAULT_SSH_PORT_MESSAGE