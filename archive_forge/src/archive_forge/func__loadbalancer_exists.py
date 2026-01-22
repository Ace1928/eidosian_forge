from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _loadbalancer_exists(self, name):
    """
        Verify a loadbalancer exists
        :param name: Name of loadbalancer
        :return: False or the ID of the existing loadbalancer
        """
    result = False
    for lb in self.lb_dict:
        if lb.get('name') == name:
            result = lb.get('id')
    return result