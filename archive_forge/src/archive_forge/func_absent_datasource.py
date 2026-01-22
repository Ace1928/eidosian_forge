from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def absent_datasource(module):
    if module.params['grafana_url'][-1] == '/':
        module.params['grafana_url'] = module.params['grafana_url'][:-1]
    api_url = module.params['grafana_url'] + '/api/datasources/name/' + module.params['dataSource']['name']
    result = requests.delete(api_url, headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
    if result.status_code == 200:
        return (False, True, {'status': result.status_code, 'response': result.json()['message']})
    else:
        return (True, False, {'status': result.status_code, 'response': result.json()['message']})