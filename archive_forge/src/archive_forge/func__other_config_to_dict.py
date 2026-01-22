from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def _other_config_to_dict(text):
    text = text.strip()
    if text == '{}':
        return None
    else:
        d = {}
        for kv in text[1:-1].split(','):
            kv = kv.strip()
            k, v = kv.split('=')
            d[k] = v
        return d