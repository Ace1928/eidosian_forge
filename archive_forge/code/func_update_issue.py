from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def update_issue(self, issue, options):
    if self._module.check_mode:
        self._module.exit_json(changed=True, msg="Successfully updated Issue '%s'." % issue['title'])
    try:
        return self.project.issues.update(issue.iid, options)
    except gitlab.exceptions.GitlabUpdateError as e:
        self._module.fail_json(msg='Failed to update Issue %s.' % to_native(e))