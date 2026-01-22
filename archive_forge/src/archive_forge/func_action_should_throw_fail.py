from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.six.moves.urllib import parse as urllib_parse
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.rabbitmq.plugins.module_utils.rabbitmq import rabbitmq_argument_spec
def action_should_throw_fail(self):
    """
        :return:
        """
    if not self.action_reply_is_correct():
        return True
    return False