from __future__ import (absolute_import, division, print_function)
import json
import re
import time
import os
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.parameters import env_fallback
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import config_ipv6
def wait_for_job_complete(self, task_uri, job_wait=False):
    """
        This function wait till the job completion.
        :param task_uri: uri to track job.
        :param job_wait: True or False decide whether to wait till the job completion.
        :return: object
        """
    response = None
    while job_wait:
        try:
            response = self.invoke_request(task_uri, 'GET')
            if response.json_data.get('TaskState') == 'Running':
                time.sleep(10)
            else:
                break
        except ValueError:
            response = response.body
            break
    return response