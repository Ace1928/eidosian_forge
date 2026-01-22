from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _define_module_argument_spec():
    """
        Define the argument spec for the ansible module
        :return: argument spec dictionary
        """
    argument_spec = dict(name=dict(), id=dict(), alias=dict(required=True), alert_recipients=dict(type='list', elements='str'), metric=dict(choices=['cpu', 'memory', 'disk']), duration=dict(type='str'), threshold=dict(type='int'), state=dict(default='present', choices=['present', 'absent']))
    mutually_exclusive = [['name', 'id']]
    return {'argument_spec': argument_spec, 'mutually_exclusive': mutually_exclusive}