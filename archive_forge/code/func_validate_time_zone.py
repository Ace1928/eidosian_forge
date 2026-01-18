from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def validate_time_zone(module, rest_obj):
    params = module.params
    time_zone = params.get('time_zone', None)
    if time_zone is not None:
        time_zone_resp = rest_obj.invoke_request('GET', TIME_ZONE)
        time_zone_val = time_zone_resp.json_data['value']
        time_id_list = [time_dict['Id'] for time_dict in time_zone_val]
        if time_zone not in time_id_list:
            sorted_time_id_list = sorted(time_id_list, key=lambda time_id: [int(i) for i in time_id.split('_') if i.isdigit()])
            module.fail_json(msg='Provide valid time zone.Choices are {0}'.format(','.join(sorted_time_id_list)))