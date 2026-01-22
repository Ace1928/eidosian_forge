from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
class GitlabBranch(object):

    def __init__(self, module, project, gitlab_instance):
        self.repo = gitlab_instance
        self._module = module
        self.project = self.get_project(project)

    def get_project(self, project):
        try:
            return self.repo.projects.get(project)
        except Exception as e:
            return False

    def get_branch(self, branch):
        try:
            return self.project.branches.get(branch)
        except Exception as e:
            return False

    def create_branch(self, branch, ref_branch):
        return self.project.branches.create({'branch': branch, 'ref': ref_branch})

    def delete_branch(self, branch):
        return branch.delete()