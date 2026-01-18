from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def update_deploy_key(self, deploy_key, arguments):
    changed = False
    for arg_key, arg_value in arguments.items():
        if arguments[arg_key] is not None:
            if getattr(deploy_key, arg_key) != arguments[arg_key]:
                setattr(deploy_key, arg_key, arguments[arg_key])
                changed = True
    return (changed, deploy_key)