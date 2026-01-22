from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import (
def _http_param_updated(self, key, resource):
    param_http = self._module.params.get('http')
    param = param_http[key]
    if param is None:
        return False
    if not resource or key not in resource['http']:
        return False
    is_different = self.find_http_difference(key, resource, param)
    if is_different:
        self._result['changed'] = True
        patch_data = {'http': {key: param}}
        before_data = {'http': {key: resource['http'][key]}}
        self._result['diff']['before'].update(before_data)
        self._result['diff']['after'].update(patch_data)
        if not self._module.check_mode:
            href = resource.get('href')
            if not href:
                self._module.fail_json(msg='Unable to update %s, no href found.' % key)
            self._patch(href, patch_data)
            return True
    return False