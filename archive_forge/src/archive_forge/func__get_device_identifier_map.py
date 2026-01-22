from __future__ import (absolute_import, division, print_function)
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def _get_device_identifier_map(module_params, rest_obj):
    """
    Builds the identifiers mapping
    :returns: the dict of device_id to server_tag map
     eg: {"device_id":{1234: None},"device_service_tag":{1345:"MXL1234"}}"""
    system_query_options_param = module_params.get('system_query_options')
    device_id_service_tag_dict = {}
    if system_query_options_param is not None:
        device_id_list = system_query_options_param.get('device_id')
        device_service_tag_list = system_query_options_param.get('device_service_tag')
        if device_id_list:
            device_id_dict = dict(((device_id, None) for device_id in list(set(device_id_list))))
            device_id_service_tag_dict['device_id'] = device_id_dict
        if device_service_tag_list:
            service_tag_dict = _get_device_id_from_service_tags(device_service_tag_list, rest_obj)
            _check_duplicate_device_id(device_id_list, service_tag_dict)
            device_id_service_tag_dict['device_service_tag'] = service_tag_dict
    return device_id_service_tag_dict