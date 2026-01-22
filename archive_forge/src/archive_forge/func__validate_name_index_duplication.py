from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import idrac_redfish_job_tracking, \
def _validate_name_index_duplication(params):
    """
    Validate for duplicate names and indices.
    :param params: Ansible list of dict
    :return: bool or error.
    """
    msg = ''
    for i in range(len(params) - 1):
        for j in range(i + 1, len(params)):
            if params[i]['Name'] == params[j]['Name']:
                msg = 'duplicate name  {0}'.format(params[i]['Name'])
                return msg
    return msg