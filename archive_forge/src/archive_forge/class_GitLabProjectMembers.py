from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.gitlab import (
class GitLabProjectMembers(object):

    def __init__(self, module, gl):
        self._module = module
        self._gitlab = gl

    def get_project(self, project_name):
        try:
            project_exists = self._gitlab.projects.get(project_name)
            return project_exists.id
        except gitlab.exceptions.GitlabGetError as e:
            project_exists = self._gitlab.projects.list(search=project_name, all=False)
            if project_exists:
                return project_exists[0].id

    def get_user_id(self, gitlab_user):
        user_exists = self._gitlab.users.list(username=gitlab_user, all=False)
        if user_exists:
            return user_exists[0].id

    def get_members_in_a_project(self, gitlab_project_id):
        project = self._gitlab.projects.get(gitlab_project_id)
        return project.members.list(all=True)

    def get_member_in_a_project(self, gitlab_project_id, gitlab_user_id):
        member = None
        project = self._gitlab.projects.get(gitlab_project_id)
        try:
            member = project.members.get(gitlab_user_id)
            if member:
                return member
        except gitlab.exceptions.GitlabGetError as e:
            return None

    def is_user_a_member(self, members, gitlab_user_id):
        for member in members:
            if member.id == gitlab_user_id:
                return True
        return False

    def add_member_to_project(self, gitlab_user_id, gitlab_project_id, access_level):
        project = self._gitlab.projects.get(gitlab_project_id)
        add_member = project.members.create({'user_id': gitlab_user_id, 'access_level': access_level})

    def remove_user_from_project(self, gitlab_user_id, gitlab_project_id):
        project = self._gitlab.projects.get(gitlab_project_id)
        project.members.delete(gitlab_user_id)

    def get_user_access_level(self, members, gitlab_user_id):
        for member in members:
            if member.id == gitlab_user_id:
                return member.access_level

    def update_user_access_level(self, members, gitlab_user_id, access_level):
        for member in members:
            if member.id == gitlab_user_id:
                member.access_level = access_level
                member.save()