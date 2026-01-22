from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.lxd import (
from ansible.module_utils.basic import AnsibleModule
import os
class LXDProjectManagement(object):

    def __init__(self, module):
        """Management of LXC projects via Ansible.

        :param module: Processed Ansible Module.
        :type module: ``object``
        """
        self.module = module
        self.name = self.module.params['name']
        self._build_config()
        self.state = self.module.params['state']
        self.new_name = self.module.params.get('new_name', None)
        self.key_file = self.module.params.get('client_key')
        if self.key_file is None:
            self.key_file = default_key_file()
        self.cert_file = self.module.params.get('client_cert')
        if self.cert_file is None:
            self.cert_file = default_cert_file()
        self.debug = self.module._verbosity >= 4
        try:
            if self.module.params['url'] != ANSIBLE_LXD_DEFAULT_URL:
                self.url = self.module.params['url']
            elif os.path.exists(self.module.params['snap_url'].replace('unix:', '')):
                self.url = self.module.params['snap_url']
            else:
                self.url = self.module.params['url']
        except Exception as e:
            self.module.fail_json(msg=e.msg)
        try:
            self.client = LXDClient(self.url, key_file=self.key_file, cert_file=self.cert_file, debug=self.debug)
        except LXDClientException as e:
            self.module.fail_json(msg=e.msg)
        self.trust_password = self.module.params.get('trust_password', None)
        self.actions = []

    def _build_config(self):
        self.config = {}
        for attr in CONFIG_PARAMS:
            param_val = self.module.params.get(attr, None)
            if param_val is not None:
                self.config[attr] = param_val

    def _get_project_json(self):
        return self.client.do('GET', '/1.0/projects/{0}'.format(self.name), ok_error_codes=[404])

    @staticmethod
    def _project_json_to_module_state(resp_json):
        if resp_json['type'] == 'error':
            return 'absent'
        return 'present'

    def _update_project(self):
        if self.state == 'present':
            if self.old_state == 'absent':
                if self.new_name is None:
                    self._create_project()
                else:
                    self.module.fail_json(msg='new_name must not be set when the project does not exist and the state is present', changed=False)
            else:
                if self.new_name is not None and self.new_name != self.name:
                    self._rename_project()
                if self._needs_to_apply_project_configs():
                    self._apply_project_configs()
        elif self.state == 'absent':
            if self.old_state == 'present':
                if self.new_name is None:
                    self._delete_project()
                else:
                    self.module.fail_json(msg='new_name must not be set when the project exists and the specified state is absent', changed=False)

    def _create_project(self):
        config = self.config.copy()
        config['name'] = self.name
        self.client.do('POST', '/1.0/projects', config)
        self.actions.append('create')

    def _rename_project(self):
        config = {'name': self.new_name}
        self.client.do('POST', '/1.0/projects/{0}'.format(self.name), config)
        self.actions.append('rename')
        self.name = self.new_name

    def _needs_to_change_project_config(self, key):
        if key not in self.config:
            return False
        old_configs = self.old_project_json['metadata'].get(key, None)
        return self.config[key] != old_configs

    def _needs_to_apply_project_configs(self):
        return self._needs_to_change_project_config('config') or self._needs_to_change_project_config('description')

    def _merge_dicts(self, source, destination):
        """ Return a new dict that merge two dict,
        with values in source dict overwrite destination dict

        Args:
            dict(source): source dict
            dict(destination): destination dict
        Kwargs:
            None
        Raises:
            None
        Returns:
            dict(destination): merged dict"""
        result = destination.copy()
        for key, value in source.items():
            if isinstance(value, dict):
                node = result.setdefault(key, {})
                self._merge_dicts(value, node)
            else:
                result[key] = value
        return result

    def _apply_project_configs(self):
        """ Selection of the procedure: rebuild or merge

        The standard behavior is that all information not contained
        in the play is discarded.

        If "merge_project" is provides in the play and "True", then existing
        configurations from the project and new ones defined are merged.

        Args:
            None
        Kwargs:
            None
        Raises:
            None
        Returns:
            None"""
        old_config = dict()
        old_metadata = self.old_project_json['metadata'].copy()
        for attr in CONFIG_PARAMS:
            old_config[attr] = old_metadata[attr]
        if self.module.params['merge_project']:
            config = self._merge_dicts(self.config, old_config)
            if config == old_config:
                return
        else:
            config = self.config.copy()
        self.client.do('PUT', '/1.0/projects/{0}'.format(self.name), config)
        self.actions.append('apply_projects_configs')

    def _delete_project(self):
        self.client.do('DELETE', '/1.0/projects/{0}'.format(self.name))
        self.actions.append('delete')

    def run(self):
        """Run the main method."""
        try:
            if self.trust_password is not None:
                self.client.authenticate(self.trust_password)
            self.old_project_json = self._get_project_json()
            self.old_state = self._project_json_to_module_state(self.old_project_json)
            self._update_project()
            state_changed = len(self.actions) > 0
            result_json = {'changed': state_changed, 'old_state': self.old_state, 'actions': self.actions}
            if self.client.debug:
                result_json['logs'] = self.client.logs
            self.module.exit_json(**result_json)
        except LXDClientException as e:
            state_changed = len(self.actions) > 0
            fail_params = {'msg': e.msg, 'changed': state_changed, 'actions': self.actions}
            if self.client.debug:
                fail_params['logs'] = e.kwargs['logs']
            self.module.fail_json(**fail_params)