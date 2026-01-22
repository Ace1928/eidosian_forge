from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def _tag_to_str(text):
    text = text.strip()
    if text == '[]':
        return None
    else:
        return text