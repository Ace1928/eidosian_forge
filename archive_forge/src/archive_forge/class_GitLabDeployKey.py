from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
class GitLabDeployKey(object):

    def __init__(self, module, gitlab_instance):
        self._module = module
        self._gitlab = gitlab_instance
        self.deploy_key_object = None
    '\n    @param project Project object\n    @param key_title Title of the key\n    @param key_key String of the key\n    @param key_can_push Option of the deploy_key\n    @param options Deploy key options\n    '

    def create_or_update_deploy_key(self, project, key_title, key_key, options):
        changed = False
        if self.deploy_key_object and self.deploy_key_object.key != key_key:
            if not self._module.check_mode:
                self.deploy_key_object.delete()
            self.deploy_key_object = None
        if self.deploy_key_object is None:
            deploy_key = self.create_deploy_key(project, {'title': key_title, 'key': key_key, 'can_push': options['can_push']})
            changed = True
        else:
            changed, deploy_key = self.update_deploy_key(self.deploy_key_object, {'title': key_title, 'can_push': options['can_push']})
        self.deploy_key_object = deploy_key
        if changed:
            if self._module.check_mode:
                self._module.exit_json(changed=True, msg='Successfully created or updated the deploy key %s' % key_title)
            try:
                deploy_key.save()
            except Exception as e:
                self._module.fail_json(msg='Failed to update deploy key: %s ' % e)
            return True
        else:
            return False
    '\n    @param project Project Object\n    @param arguments Attributes of the deploy_key\n    '

    def create_deploy_key(self, project, arguments):
        if self._module.check_mode:
            return True
        try:
            deploy_key = project.keys.create(arguments)
        except gitlab.exceptions.GitlabCreateError as e:
            self._module.fail_json(msg='Failed to create deploy key: %s ' % to_native(e))
        return deploy_key
    '\n    @param deploy_key Deploy Key Object\n    @param arguments Attributes of the deploy_key\n    '

    def update_deploy_key(self, deploy_key, arguments):
        changed = False
        for arg_key, arg_value in arguments.items():
            if arguments[arg_key] is not None:
                if getattr(deploy_key, arg_key) != arguments[arg_key]:
                    setattr(deploy_key, arg_key, arguments[arg_key])
                    changed = True
        return (changed, deploy_key)
    '\n    @param project Project object\n    @param key_title Title of the key\n    '

    def find_deploy_key(self, project, key_title):
        for deploy_key in project.keys.list(**list_all_kwargs):
            if deploy_key.title == key_title:
                return deploy_key
    '\n    @param project Project object\n    @param key_title Title of the key\n    '

    def exists_deploy_key(self, project, key_title):
        deploy_key = self.find_deploy_key(project, key_title)
        if deploy_key:
            self.deploy_key_object = deploy_key
            return True
        return False

    def delete_deploy_key(self):
        if self._module.check_mode:
            return True
        return self.deploy_key_object.delete()