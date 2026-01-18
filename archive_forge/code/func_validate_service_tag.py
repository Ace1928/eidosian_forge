from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def validate_service_tag(device_service_tag, identifier, device_type_map, rest_obj, module):
    """
    Validate the service tag and device type of device
    :param identifier: identifier options which required find service tag from module params
    primary_switch_service_tag, secondary_switch_service_tag, hostname
    :param device_service_tag: device service tag
    :param device_type_map: map to get the
    :param rest_obj: session object
    :param module: ansible module object
    :return: None
    """
    if device_service_tag is not None:
        device_id_details = rest_obj.get_device_id_from_service_tag(device_service_tag)
        device_details = device_id_details['value']
        if device_id_details['Id'] is None:
            module.fail_json(msg=DEVICE_SERVICE_TAG_NOT_FOUND_ERROR_MSG.format(device_service_tag))
        identifier_device_type = device_details['Type']
        validate_device_type(device_type_map[identifier_device_type], identifier, device_details, module)