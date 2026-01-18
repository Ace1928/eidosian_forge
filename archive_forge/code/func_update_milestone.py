from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
from datetime import datetime
def update_milestone(self, var_obj):
    if self._module.check_mode:
        return (True, True)
    _milestone = self.gitlab_object.milestones.get(self.get_milestone_id(var_obj.get('title')))
    if var_obj.get('description') is not None:
        _milestone.description = var_obj.get('description')
    if var_obj.get('start_date') is not None:
        _milestone.start_date = var_obj.get('start_date')
    if var_obj.get('due_date') is not None:
        _milestone.due_date = var_obj.get('due_date')
    _milestone.save()
    return (True, _milestone.asdict())