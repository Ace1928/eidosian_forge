from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _create_clc_server(clc, module, server_params):
    """
        Call the CLC Rest API to Create a Server
        :param clc: the clc-python-sdk instance to use
        :param module: the AnsibleModule instance to use
        :param server_params: a dictionary of params to use to create the servers
        :return: clc-sdk.Request object linked to the queued server request
        """
    try:
        res = clc.v2.API.Call(method='POST', url='servers/%s' % server_params.get('alias'), payload=json.dumps({'name': server_params.get('name'), 'description': server_params.get('description'), 'groupId': server_params.get('group_id'), 'sourceServerId': server_params.get('template'), 'isManagedOS': server_params.get('managed_os'), 'primaryDNS': server_params.get('primary_dns'), 'secondaryDNS': server_params.get('secondary_dns'), 'networkId': server_params.get('network_id'), 'ipAddress': server_params.get('ip_address'), 'password': server_params.get('password'), 'sourceServerPassword': server_params.get('source_server_password'), 'cpu': server_params.get('cpu'), 'cpuAutoscalePolicyId': server_params.get('cpu_autoscale_policy_id'), 'memoryGB': server_params.get('memory'), 'type': server_params.get('type'), 'storageType': server_params.get('storage_type'), 'antiAffinityPolicyId': server_params.get('anti_affinity_policy_id'), 'customFields': server_params.get('custom_fields'), 'additionalDisks': server_params.get('additional_disks'), 'ttl': server_params.get('ttl'), 'packages': server_params.get('packages'), 'configurationId': server_params.get('configuration_id'), 'osType': server_params.get('os_type')}))
        result = clc.v2.Requests(res)
    except APIFailedResponse as ex:
        return module.fail_json(msg='Unable to create the server: {0}. {1}'.format(server_params.get('name'), ex.response_text))
    server_uuid = [obj['id'] for obj in res['links'] if obj['rel'] == 'self'][0]
    result.requests[0].Server = lambda: ClcServer._find_server_by_uuid_w_retry(clc, module, server_uuid, server_params.get('alias'))
    return result