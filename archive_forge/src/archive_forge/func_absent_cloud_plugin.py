from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def absent_cloud_plugin(module):
    api_url = 'https://grafana.com/api/instances/' + module.params['stack_slug'] + '/plugins/' + module.params['name']
    result = requests.delete(api_url, headers={'Authorization': 'Bearer ' + module.params['cloud_api_key']})
    if result.status_code == 200:
        return (False, True, result.json())
    else:
        return (True, False, {'status': result.status_code, 'response': result.json()['message']})