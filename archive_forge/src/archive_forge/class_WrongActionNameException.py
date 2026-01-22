from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
class WrongActionNameException(RunnableException):

    def __init__(self, action, available_actions):
        super(WrongActionNameException, self).__init__('Wrong action name ' + repr(action), 'Available actions are: ' + repr(available_actions))