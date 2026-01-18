from __future__ import absolute_import, division, print_function
import base64
import hashlib
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def user_disable(self, name):
    return self._post_json(method='user_disable', name=name)