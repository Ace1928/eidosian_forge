from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def validate_lead_msm_version(each_domain, module, fabric_design=None):
    """
    validate lead chassis for design type
    and find the msm version of the domain
    """
    role_type = each_domain['DomainRoleTypeValue'].upper()
    if fabric_design and fabric_design == '2xMX9116n_Fabric_Switching_Engines_in_different_chassis' and (role_type != 'LEAD'):
        module.fail_json(msg=LEAD_CHASSIS_ERROR_MSG.format(fabric_design))
    msm_version = each_domain['Version']
    return msm_version