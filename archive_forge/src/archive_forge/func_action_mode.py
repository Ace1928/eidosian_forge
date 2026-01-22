from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.six.moves.urllib import parse as urllib_parse
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.rabbitmq.plugins.module_utils.rabbitmq import rabbitmq_argument_spec
def action_mode(self):
    """
        :return:
        """
    result = self.result
    if self.change_required():
        if self.module.params['state'] == 'present':
            self.create()
        if self.module.params['state'] == 'absent':
            self.remove()
        if self.action_should_throw_fail():
            self.fail()
        result['changed'] = True
        result['destination'] = self.module.params['destination']
        self.module.exit_json(**result)
    else:
        result['changed'] = False
        self.module.exit_json(**result)