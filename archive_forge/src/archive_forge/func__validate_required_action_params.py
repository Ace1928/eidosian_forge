from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
def _validate_required_action_params(self):
    action_params_names = list(self.action_params)
    res = self.client.runnable.get_action_info(self.type, self.name, self.action_name)
    if 'parameters' in res[rest_client.RESP_DATA]:
        action_params_spec = res[rest_client.RESP_DATA]['parameters']
    else:
        action_params_spec = []
    required_action_params_dict = [{'name': p['name'], 'type': p['type']} for p in action_params_spec if p['type'] == 'alteon' or p['type'] == 'defensePro' or p['type'] == 'appWall' or (p['type'] == 'alteon[]') or (p['type'] == 'defensePro[]') or (p['type'] == 'appWall[]') or (p['direction'] != 'out')]
    required_action_params_names = [n['name'] for n in required_action_params_dict]
    if set(required_action_params_names) & set(action_params_names) != set(required_action_params_names):
        raise MissingActionParametersException(required_action_params_dict)