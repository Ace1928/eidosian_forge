from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
class RunnableException(Exception):

    def __init__(self, reason, details):
        self.reason = reason
        self.details = details

    def __str__(self):
        return 'Reason: {0}. Details:{1}.'.format(self.reason, self.details)