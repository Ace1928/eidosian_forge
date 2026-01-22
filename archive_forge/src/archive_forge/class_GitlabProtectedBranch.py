from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
class GitlabProtectedBranch(object):

    def __init__(self, module, project, gitlab_instance):
        self.repo = gitlab_instance
        self._module = module
        self.project = self.get_project(project)
        self.ACCESS_LEVEL = {'nobody': gitlab.const.NO_ACCESS, 'developer': gitlab.const.DEVELOPER_ACCESS, 'maintainer': gitlab.const.MAINTAINER_ACCESS}

    def get_project(self, project_name):
        return self.repo.projects.get(project_name)

    def protected_branch_exist(self, name):
        try:
            return self.project.protectedbranches.get(name)
        except Exception as e:
            return False

    def create_protected_branch(self, name, merge_access_levels, push_access_level):
        if self._module.check_mode:
            return True
        merge = self.ACCESS_LEVEL[merge_access_levels]
        push = self.ACCESS_LEVEL[push_access_level]
        self.project.protectedbranches.create({'name': name, 'merge_access_level': merge, 'push_access_level': push})

    def compare_protected_branch(self, name, merge_access_levels, push_access_level):
        configured_merge = self.ACCESS_LEVEL[merge_access_levels]
        configured_push = self.ACCESS_LEVEL[push_access_level]
        current = self.protected_branch_exist(name=name)
        current_merge = current.merge_access_levels[0]['access_level']
        current_push = current.push_access_levels[0]['access_level']
        if current:
            if current.name == name and current_merge == configured_merge and (current_push == configured_push):
                return True
        return False

    def delete_protected_branch(self, name):
        if self._module.check_mode:
            return True
        return self.project.protectedbranches.delete(name)