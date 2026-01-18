from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def wait_action(self, droplet_id, desired_action_data):
    action_type = desired_action_data.get('type', 'undefined')
    response = self.rest.post('droplets/{0}/actions'.format(droplet_id), data=desired_action_data)
    json_data = response.json
    status_code = response.status_code
    message = json_data.get('message', 'no error message')
    action = json_data.get('action', None)
    action_id = action.get('id', None)
    action_status = action.get('status', None)
    if action is None or action_id is None or action_status is None:
        self.module.fail_json(changed=False, msg=DODroplet.failure_message['unexpected'].format('no action, ID, or status'))
    if status_code >= 400:
        self.module.fail_json(changed=False, msg=DODroplet.failure_message['failed_to'].format('post', 'action', status_code, message))
    self.wait_check_action(droplet_id, action_id)