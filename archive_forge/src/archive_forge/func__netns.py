from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
def _netns(self, command):
    """Run ip nents command"""
    return self.module.run_command(['ip', 'netns'] + command)