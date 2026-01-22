from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib

        Checks if the alert policy exists for the server
        :param server: the clc server object
        :param alert_policy_id: the alert policy
        :return: True: if the given alert policy id associated to the server, False otherwise
        