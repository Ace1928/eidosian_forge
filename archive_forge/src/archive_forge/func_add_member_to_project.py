from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def add_member_to_project(self, gitlab_user_id, gitlab_project_id, access_level):
    project = self._gitlab.projects.get(gitlab_project_id)
    add_member = project.members.create({'user_id': gitlab_user_id, 'access_level': access_level})