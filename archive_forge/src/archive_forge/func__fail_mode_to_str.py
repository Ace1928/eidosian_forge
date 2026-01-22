from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def _fail_mode_to_str(text):
    if not text:
        return None
    else:
        return text.strip()