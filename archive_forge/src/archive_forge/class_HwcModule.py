from __future__ import (absolute_import, division, print_function)
import re
import time
import traceback
from ansible.module_utils.basic import (AnsibleModule, env_fallback,
from ansible.module_utils.common.text.converters import to_text
class HwcModule(AnsibleModule):

    def __init__(self, *args, **kwargs):
        arg_spec = kwargs.setdefault('argument_spec', {})
        arg_spec.update(dict(identity_endpoint=dict(required=True, type='str', fallback=(env_fallback, ['ANSIBLE_HWC_IDENTITY_ENDPOINT'])), user=dict(required=True, type='str', fallback=(env_fallback, ['ANSIBLE_HWC_USER'])), password=dict(required=True, type='str', no_log=True, fallback=(env_fallback, ['ANSIBLE_HWC_PASSWORD'])), domain=dict(required=True, type='str', fallback=(env_fallback, ['ANSIBLE_HWC_DOMAIN'])), project=dict(required=True, type='str', fallback=(env_fallback, ['ANSIBLE_HWC_PROJECT'])), region=dict(type='str', fallback=(env_fallback, ['ANSIBLE_HWC_REGION'])), id=dict(type='str')))
        super(HwcModule, self).__init__(*args, **kwargs)