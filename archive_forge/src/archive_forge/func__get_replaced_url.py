from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def _get_replaced_url(self, url_template):
    target_url = url_template
    for param in self.url_params:
        token_hint = '{%s}' % param
        token = ''
        modified_name = _get_modified_name(param)
        modified_token = self.module.params.get(modified_name, None)
        previous_token = self.module.params.get(param, None)
        if modified_token is not None:
            token = modified_token
        elif previous_token is not None:
            token = previous_token
        else:
            self.module.fail_json(msg='Missing input param: %s' % modified_name)
        target_url = target_url.replace(token_hint, '%s' % token)
    return target_url