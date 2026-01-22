from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.module_utils.six.moves.urllib.parse import urlencode
class LXDProfileManagement(object):

    def __init__(self, module):
        """Management of LXC profiles via Ansible.

        :param module: Processed Ansible Module.
        :type module: ``object``
        """
        self.module = module
        self.name = self.module.params['name']
        self.project = self.module.params['project']
        self._build_config()
        self.state = self.module.params['state']
        self.new_name = self.module.params.get('new_name', None)
        self.key_file = self.module.params.get('client_key')
        if self.key_file is None:
            self.key_file = '{0}/.config/lxc/client.key'.format(os.environ['HOME'])
        self.cert_file = self.module.params.get('client_cert')
        if self.cert_file is None:
            self.cert_file = '{0}/.config/lxc/client.crt'.format(os.environ['HOME'])
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

    def _get_profile_json(self):
        url = '/1.0/profiles/{0}'.format(self.name)
        if self.project:
            url = '{0}?{1}'.format(url, urlencode(dict(project=self.project)))
        return self.client.do('GET', url, ok_error_codes=[404])

    @staticmethod
    def _profile_json_to_module_state(resp_json):
        if resp_json['type'] == 'error':
            return 'absent'
        return 'present'

    def _update_profile(self):
        if self.state == 'present':
            if self.old_state == 'absent':
                if self.new_name is None:
                    self._create_profile()
                else:
                    self.module.fail_json(msg='new_name must not be set when the profile does not exist and the state is present', changed=False)
            else:
                if self.new_name is not None and self.new_name != self.name:
                    self._rename_profile()
                if self._needs_to_apply_profile_configs():
                    self._apply_profile_configs()
        elif self.state == 'absent':
            if self.old_state == 'present':
                if self.new_name is None:
                    self._delete_profile()
                else:
                    self.module.fail_json(msg='new_name must not be set when the profile exists and the specified state is absent', changed=False)

    def _create_profile(self):
        url = '/1.0/profiles'
        if self.project:
            url = '{0}?{1}'.format(url, urlencode(dict(project=self.project)))
        config = self.config.copy()
        config['name'] = self.name
        self.client.do('POST', url, config)
        self.actions.append('create')

    def _rename_profile(self):
        url = '/1.0/profiles/{0}'.format(self.name)
        if self.project:
            url = '{0}?{1}'.format(url, urlencode(dict(project=self.project)))
        config = {'name': self.new_name}
        self.client.do('POST', url, config)
        self.actions.append('rename')
        self.name = self.new_name

    def _needs_to_change_profile_config(self, key):
        if key not in self.config:
            return False
        old_configs = self.old_profile_json['metadata'].get(key, None)
        return self.config[key] != old_configs

    def _needs_to_apply_profile_configs(self):
        return self._needs_to_change_profile_config('config') or self._needs_to_change_profile_config('description') or self._needs_to_change_profile_config('devices')

    def _merge_dicts(self, source, destination):
        """Merge Dictionaries

        Get a list of filehandle numbers from logger to be handed to
        DaemonContext.files_preserve

        Args:
            dict(source): source dict
            dict(destination): destination dict
        Kwargs:
            None
        Raises:
            None
        Returns:
            dict(destination): merged dict"""
        for key, value in source.items():
            if isinstance(value, dict):
                node = destination.setdefault(key, {})
                self._merge_dicts(value, node)
            else:
                destination[key] = value
        return destination

    def _merge_config(self, config):
        """ merge profile

        Merge Configuration of the present profile and the new desired configitems

        Args:
            dict(config): Dict with the old config in 'metadata' and new config in 'config'
        Kwargs:
            None
        Raises:
            None
        Returns:
            dict(config): new config"""
        for item in ['config', 'description', 'devices', 'name', 'used_by']:
            if item in config:
                config[item] = self._merge_dicts(config['metadata'][item], config[item])
            else:
                config[item] = config['metadata'][item]
        return self._merge_dicts(self.config, config)

    def _generate_new_config(self, config):
        """ rebuild profile

        Rebuild the Profile by the configuration provided in the play.
        Existing configurations are discarded.

        This is the default behavior.

        Args:
            dict(config): Dict with the old config in 'metadata' and new config in 'config'
        Kwargs:
            None
        Raises:
            None
        Returns:
            dict(config): new config"""
        for k, v in self.config.items():
            config[k] = v
        return config

    def _apply_profile_configs(self):
        """ Selection of the procedure: rebuild or merge

        The standard behavior is that all information not contained
        in the play is discarded.

        If "merge_profile" is provides in the play and "True", then existing
        configurations from the profile and new ones defined are merged.

        Args:
            None
        Kwargs:
            None
        Raises:
            None
        Returns:
            None"""
        config = self.old_profile_json.copy()
        if self.module.params['merge_profile']:
            config = self._merge_config(config)
        else:
            config = self._generate_new_config(config)
        url = '/1.0/profiles/{0}'.format(self.name)
        if self.project:
            url = '{0}?{1}'.format(url, urlencode(dict(project=self.project)))
        self.client.do('PUT', url, config)
        self.actions.append('apply_profile_configs')

    def _delete_profile(self):
        url = '/1.0/profiles/{0}'.format(self.name)
        if self.project:
            url = '{0}?{1}'.format(url, urlencode(dict(project=self.project)))
        self.client.do('DELETE', url)
        self.actions.append('delete')

    def run(self):
        """Run the main method."""
        try:
            if self.trust_password is not None:
                self.client.authenticate(self.trust_password)
            self.old_profile_json = self._get_profile_json()
            self.old_state = self._profile_json_to_module_state(self.old_profile_json)
            self._update_profile()
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