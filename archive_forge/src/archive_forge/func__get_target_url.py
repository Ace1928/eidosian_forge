from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def _get_target_url(self, adom_value, url_list):
    target_url = None
    if adom_value is not None and (not url_list[0].endswith('{adom}')):
        if adom_value == 'global':
            for url in url_list:
                if '/global/' in url and '/adom/{adom}/' not in url:
                    target_url = url
                    break
        elif adom_value:
            for url in url_list:
                if '/adom/{adom}/' in url:
                    target_url = url
                    break
        else:
            for url in url_list:
                if '/global/' not in url and '/adom/{adom}/' not in url:
                    target_url = url
                    break
    else:
        target_url = url_list[0]
    if not target_url:
        self.module.fail_json(msg='can not find url in following sets:%s! please check params: adom' % target_url)
    return target_url