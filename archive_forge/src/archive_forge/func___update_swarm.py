from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils.swarm import AnsibleDockerSwarmClient
from ansible.module_utils.common.text.converters import to_native
def __update_swarm(self):
    try:
        self.inspect_swarm()
        version = self.swarm_info['Version']['Index']
        self.parameters.update_from_swarm_info(self.swarm_info)
        old_parameters = TaskParameters()
        old_parameters.update_from_swarm_info(self.swarm_info)
        self.parameters.compare_to_active(old_parameters, self.client, self.differences)
        if self.differences.empty:
            self.results['actions'].append('No modification')
            self.results['changed'] = False
            return
        update_parameters = TaskParameters.from_ansible_params(self.client)
        update_parameters.update_parameters(self.client)
        if not self.check_mode:
            self.client.update_swarm(version=version, swarm_spec=update_parameters.spec, rotate_worker_token=self.parameters.rotate_worker_token, rotate_manager_token=self.parameters.rotate_manager_token)
    except APIError as exc:
        self.client.fail('Can not update a Swarm Cluster: %s' % to_native(exc))
        return
    self.inspect_swarm()
    self.results['actions'].append('Swarm cluster updated')
    self.results['changed'] = True