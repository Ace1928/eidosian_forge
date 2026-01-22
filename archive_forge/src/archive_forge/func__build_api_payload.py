from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils import arguments, errors, utils
def _build_api_payload(client, params):
    payload = arguments.get_spec_payload(params, 'timestamp')
    payload['metadata'] = dict(namespace=params['namespace'])
    payload['entity'] = get_entity(client, params['namespace'], params['entity'])
    payload['check'] = get_check(client, params['namespace'], params['check'])
    _update_payload_with_check_attributes(payload, params['check_attributes'])
    _update_payload_with_metric_attributes(payload, params['metric_attributes'])
    return payload