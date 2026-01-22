from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _wait_for_requests(module, request_list):
    """
        Block until server provisioning requests are completed.
        :param module: the AnsibleModule object
        :param request_list: a list of clc-sdk.Request instances
        :return: none
        """
    wait = module.params.get('wait')
    if wait:
        failed_requests_count = sum([request.WaitUntilComplete() for request in request_list])
        if failed_requests_count > 0:
            module.fail_json(msg='Unable to process server request')