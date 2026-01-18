from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def update_mr(self, mr, options):
    if self._module.check_mode:
        self._module.exit_json(changed=True, msg='Successfully updated the Merge Request %s' % mr['title'])
    try:
        return self.project.mergerequests.update(mr.iid, options)
    except gitlab.exceptions.GitlabUpdateError as e:
        self._module.fail_json(msg='Failed to update Merge Request: %s ' % to_native(e))