from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.six.moves.urllib.parse import urlparse
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _wait_for_requests_to_complete(self, source_account_alias, location, firewall_policy_id, wait_limit=50):
    """
        Waits until the CLC requests are complete if the wait argument is True
        :param source_account_alias: The source account alias for the firewall policy
        :param location: datacenter of the firewall policy
        :param firewall_policy_id: The firewall policy id
        :param wait_limit: The number of times to check the status for completion
        :return: the firewall_policy object
        """
    wait = self.module.params.get('wait')
    count = 0
    firewall_policy = None
    while wait:
        count += 1
        firewall_policy = self._get_firewall_policy(source_account_alias, location, firewall_policy_id)
        status = firewall_policy.get('status')
        if status == 'active' or count > wait_limit:
            wait = False
        else:
            sleep(2)
    return firewall_policy