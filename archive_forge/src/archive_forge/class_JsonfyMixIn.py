from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, env_fallback
class JsonfyMixIn(object):

    def to_json(self):
        return self.__dict__