from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
class GitLabGroup(object):

    def __init__(self, module, gitlab_instance):
        self._module = module
        self._gitlab = gitlab_instance
        self.group_object = None
    '\n    @param group Group object\n    '

    def get_group_id(self, group):
        if group is not None:
            return group.id
        return None
    '\n    @param name Name of the group\n    @param parent Parent group full path\n    @param options Group options\n    '

    def create_or_update_group(self, name, parent, options):
        changed = False
        if self.group_object is None:
            parent_id = self.get_group_id(parent)
            payload = {'name': name, 'path': options['path'], 'parent_id': parent_id, 'visibility': options['visibility'], 'project_creation_level': options['project_creation_level'], 'auto_devops_enabled': options['auto_devops_enabled'], 'subgroup_creation_level': options['subgroup_creation_level']}
            if options.get('description'):
                payload['description'] = options['description']
            if options.get('require_two_factor_authentication'):
                payload['require_two_factor_authentication'] = options['require_two_factor_authentication']
            group = self.create_group(payload)
            if options['avatar_path']:
                try:
                    group.avatar = open(options['avatar_path'], 'rb')
                except IOError as e:
                    self._module.fail_json(msg='Cannot open {0}: {1}'.format(options['avatar_path'], e))
            changed = True
        else:
            changed, group = self.update_group(self.group_object, {'name': name, 'description': options['description'], 'visibility': options['visibility'], 'project_creation_level': options['project_creation_level'], 'auto_devops_enabled': options['auto_devops_enabled'], 'subgroup_creation_level': options['subgroup_creation_level'], 'require_two_factor_authentication': options['require_two_factor_authentication']})
        self.group_object = group
        if changed:
            if self._module.check_mode:
                self._module.exit_json(changed=True, msg='Successfully created or updated the group %s' % name)
            try:
                group.save()
            except Exception as e:
                self._module.fail_json(msg='Failed to update group: %s ' % e)
            return True
        else:
            return False
    '\n    @param arguments Attributes of the group\n    '

    def create_group(self, arguments):
        if self._module.check_mode:
            return True
        try:
            filtered = dict(((arg_key, arg_value) for arg_key, arg_value in arguments.items() if arg_value is not None))
            group = self._gitlab.groups.create(filtered)
        except gitlab.exceptions.GitlabCreateError as e:
            self._module.fail_json(msg='Failed to create group: %s ' % to_native(e))
        return group
    '\n    @param group Group Object\n    @param arguments Attributes of the group\n    '

    def update_group(self, group, arguments):
        changed = False
        for arg_key, arg_value in arguments.items():
            if arguments[arg_key] is not None:
                if getattr(group, arg_key) != arguments[arg_key]:
                    setattr(group, arg_key, arguments[arg_key])
                    changed = True
        return (changed, group)
    '\n    @param force To delete even if projects inside\n    '

    def delete_group(self, force=False):
        group = self.group_object
        if not force and len(group.projects.list(all=False)) >= 1:
            self._module.fail_json(msg="There are still projects in this group. These needs to be moved or deleted before this group can be removed. Use 'force_delete' to 'true' to force deletion of existing projects.")
        else:
            if self._module.check_mode:
                return True
            try:
                group.delete()
            except Exception as e:
                self._module.fail_json(msg='Failed to delete group: %s ' % to_native(e))
    '\n    @param name Name of the group\n    @param full_path Complete path of the Group including parent group path. <parent_path>/<group_path>\n    '

    def exists_group(self, project_identifier):
        group = find_group(self._gitlab, project_identifier)
        if group:
            self.group_object = group
            return True
        return False