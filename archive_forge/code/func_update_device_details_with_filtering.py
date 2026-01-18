from __future__ import (absolute_import, division, print_function)
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def update_device_details_with_filtering(missing_service_tags, service_tag_dict, rest_obj):
    """
    This is a workaround solutions.
    Use filtering query, in case fetching all report list fails for some reason.
    Updates service_tag_dict if filtering request is success.
    :param missing_service_tags:  Service tags which are unable to fetch from pagination request.
    :param service_tag_dict: this contains device id mapping with tags
    :param rest_obj: ome connection object
    :return: None.
    """
    try:
        for tag in missing_service_tags:
            query = "DeviceServiceTag eq '{0}'".format(tag)
            query_param = {'$filter': query}
            resp = rest_obj.invoke_request('GET', DEVICE_RESOURCE_COLLECTION[DEVICE_LIST]['resource'], query_param=query_param)
            value = resp.json_data['value']
            if value and value[0]['DeviceServiceTag'] == tag:
                service_tag_dict.update({value[0]['Id']: value[0]['DeviceServiceTag']})
                missing_service_tags.remove(tag)
    except (URLError, HTTPError, SSLValidationError, ConnectionError, TypeError, ValueError) as err:
        raise err