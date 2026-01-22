from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
class GitlabIssue(object):

    def __init__(self, module, project, gitlab_instance):
        self._gitlab = gitlab_instance
        self._module = module
        self.project = project
    '\n    @param milestone_id Title of the milestone\n    '

    def get_milestone(self, milestone_id, group):
        milestones = []
        try:
            milestones = group.milestones.list(search=milestone_id)
        except gitlab.exceptions.GitlabGetError as e:
            self._module.fail_json(msg='Failed to list the Milestones: %s' % to_native(e))
        if len(milestones) > 1:
            self._module.fail_json(msg='Multiple Milestones matched search criteria.')
        if len(milestones) < 1:
            self._module.fail_json(msg='No Milestones matched search criteria.')
        if len(milestones) == 1:
            try:
                return group.milestones.get(id=milestones[0].id)
            except gitlab.exceptions.GitlabGetError as e:
                self._module.fail_json(msg='Failed to get the Milestones: %s' % to_native(e))
    "\n    @param title Title of the Issue\n    @param state_filter Issue's state to filter on\n    "

    def get_issue(self, title, state_filter):
        issues = []
        try:
            issues = self.project.issues.list(query_parameters={'search': title, 'in': 'title', 'state': state_filter})
        except gitlab.exceptions.GitlabGetError as e:
            self._module.fail_json(msg='Failed to list the Issues: %s' % to_native(e))
        if len(issues) > 1:
            self._module.fail_json(msg='Multiple Issues matched search criteria.')
        if len(issues) == 1:
            try:
                return self.project.issues.get(id=issues[0].iid)
            except gitlab.exceptions.GitlabGetError as e:
                self._module.fail_json(msg='Failed to get the Issue: %s' % to_native(e))
    '\n    @param username Name of the user\n    '

    def get_user(self, username):
        users = []
        try:
            users = [user for user in self.project.users.list(username=username, all=True) if user.username == username]
        except gitlab.exceptions.GitlabGetError as e:
            self._module.fail_json(msg='Failed to list the users: %s' % to_native(e))
        if len(users) > 1:
            self._module.fail_json(msg='Multiple Users matched search criteria.')
        elif len(users) < 1:
            self._module.fail_json(msg='No User matched search criteria.')
        else:
            return users[0]
    '\n    @param users List of usernames\n    '

    def get_user_ids(self, users):
        return [self.get_user(user).id for user in users]
    '\n    @param options Options of the Issue\n    '

    def create_issue(self, options):
        if self._module.check_mode:
            self._module.exit_json(changed=True, msg="Successfully created Issue '%s'." % options['title'])
        try:
            return self.project.issues.create(options)
        except gitlab.exceptions.GitlabCreateError as e:
            self._module.fail_json(msg='Failed to create Issue: %s ' % to_native(e))
    '\n    @param issue Issue object to delete\n    '

    def delete_issue(self, issue):
        if self._module.check_mode:
            self._module.exit_json(changed=True, msg="Successfully deleted Issue '%s'." % issue['title'])
        try:
            return issue.delete()
        except gitlab.exceptions.GitlabDeleteError as e:
            self._module.fail_json(msg="Failed to delete Issue: '%s'." % to_native(e))
    '\n    @param issue Issue object to update\n    @param options Options of the Issue\n    '

    def update_issue(self, issue, options):
        if self._module.check_mode:
            self._module.exit_json(changed=True, msg="Successfully updated Issue '%s'." % issue['title'])
        try:
            return self.project.issues.update(issue.iid, options)
        except gitlab.exceptions.GitlabUpdateError as e:
            self._module.fail_json(msg='Failed to update Issue %s.' % to_native(e))
    '\n    @param issue Issue object to evaluate\n    @param options New options to update Issue with\n    '

    def issue_has_changed(self, issue, options):
        for key, value in options.items():
            if value is not None:
                if key == 'milestone_id':
                    old_milestone = getattr(issue, 'milestone')['id'] if getattr(issue, 'milestone') else ''
                    if options[key] != old_milestone:
                        return True
                elif key == 'assignee_ids':
                    if options[key] != sorted([user['id'] for user in getattr(issue, 'assignees')]):
                        return True
                elif key == 'labels':
                    if options[key] != sorted(getattr(issue, key)):
                        return True
                elif getattr(issue, key) != value:
                    return True
        return False