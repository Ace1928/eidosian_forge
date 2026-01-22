from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api import auth
from ansible_collections.community.docker.plugins.module_utils._api.errors import APIError, DockerException, NotFound
class DockerPluginManager(object):

    def __init__(self, client):
        self.client = client
        self.parameters = TaskParameters(client)
        self.preferred_name = self.parameters.alias or self.parameters.plugin_name
        self.check_mode = self.client.check_mode
        self.diff = self.client.module._diff
        self.diff_tracker = DifferenceTracker()
        self.diff_result = dict()
        self.actions = []
        self.changed = False
        self.existing_plugin = self.get_existing_plugin()
        state = self.parameters.state
        if state == 'present':
            self.present()
        elif state == 'absent':
            self.absent()
        elif state == 'enable':
            self.enable()
        elif state == 'disable':
            self.disable()
        if self.diff or self.check_mode or self.parameters.debug:
            if self.diff:
                self.diff_result['before'], self.diff_result['after'] = self.diff_tracker.get_before_after()
            self.diff = self.diff_result

    def get_existing_plugin(self):
        try:
            return self.client.get_json('/plugins/{0}/json', self.preferred_name)
        except NotFound:
            return None
        except APIError as e:
            self.client.fail(to_native(e))

    def has_different_config(self):
        """
        Return the list of differences between the current parameters and the existing plugin.

        :return: list of options that differ
        """
        differences = DifferenceTracker()
        if self.parameters.plugin_options:
            settings = self.existing_plugin.get('Settings')
            if not settings:
                differences.add('plugin_options', parameters=self.parameters.plugin_options, active=settings)
            else:
                existing_options = parse_options(settings.get('Env'))
                for key, value in self.parameters.plugin_options.items():
                    if not existing_options.get(key) and value or not value or value != existing_options[key]:
                        differences.add('plugin_options.%s' % key, parameter=value, active=existing_options.get(key))
        return differences

    def install_plugin(self):
        if not self.existing_plugin:
            if not self.check_mode:
                try:
                    headers = {}
                    registry, repo_name = auth.resolve_repository_name(self.parameters.plugin_name)
                    header = auth.get_config_header(self.client, registry)
                    if header:
                        headers['X-Registry-Auth'] = header
                    privileges = self.client.get_json('/plugins/privileges', params={'remote': self.parameters.plugin_name}, headers=headers)
                    params = {'remote': self.parameters.plugin_name}
                    if self.parameters.alias:
                        params['name'] = self.parameters.alias
                    response = self.client._post_json(self.client._url('/plugins/pull'), params=params, headers=headers, data=privileges, stream=True)
                    self.client._raise_for_status(response)
                    for data in self.client._stream_helper(response, decode=True):
                        pass
                    self.existing_plugin = self.client.get_json('/plugins/{0}/json', self.preferred_name)
                    if self.parameters.plugin_options:
                        data = prepare_options(self.parameters.plugin_options)
                        self.client.post_json('/plugins/{0}/set', self.preferred_name, data=data)
                except APIError as e:
                    self.client.fail(to_native(e))
            self.actions.append('Installed plugin %s' % self.preferred_name)
            self.changed = True

    def remove_plugin(self):
        force = self.parameters.force_remove
        if self.existing_plugin:
            if not self.check_mode:
                try:
                    self.client.delete_call('/plugins/{0}', self.preferred_name, params={'force': force})
                except APIError as e:
                    self.client.fail(to_native(e))
            self.actions.append('Removed plugin %s' % self.preferred_name)
            self.changed = True

    def update_plugin(self):
        if self.existing_plugin:
            differences = self.has_different_config()
            if not differences.empty:
                if not self.check_mode:
                    try:
                        data = prepare_options(self.parameters.plugin_options)
                        self.client.post_json('/plugins/{0}/set', self.preferred_name, data=data)
                    except APIError as e:
                        self.client.fail(to_native(e))
                self.actions.append('Updated plugin %s settings' % self.preferred_name)
                self.changed = True
        else:
            self.client.fail('Cannot update the plugin: Plugin does not exist')

    def present(self):
        differences = DifferenceTracker()
        if self.existing_plugin:
            differences = self.has_different_config()
        self.diff_tracker.add('exists', parameter=True, active=self.existing_plugin is not None)
        if self.existing_plugin:
            self.update_plugin()
        else:
            self.install_plugin()
        if self.diff or self.check_mode or self.parameters.debug:
            self.diff_tracker.merge(differences)
        if not self.check_mode and (not self.parameters.debug):
            self.actions = None

    def absent(self):
        self.remove_plugin()

    def enable(self):
        timeout = self.parameters.enable_timeout
        if self.existing_plugin:
            if not self.existing_plugin.get('Enabled'):
                if not self.check_mode:
                    try:
                        self.client.post_json('/plugins/{0}/enable', self.preferred_name, params={'timeout': timeout})
                    except APIError as e:
                        self.client.fail(to_native(e))
                self.actions.append('Enabled plugin %s' % self.preferred_name)
                self.changed = True
        else:
            self.install_plugin()
            if not self.check_mode:
                try:
                    self.client.post_json('/plugins/{0}/enable', self.preferred_name, params={'timeout': timeout})
                except APIError as e:
                    self.client.fail(to_native(e))
            self.actions.append('Enabled plugin %s' % self.preferred_name)
            self.changed = True

    def disable(self):
        if self.existing_plugin:
            if self.existing_plugin.get('Enabled'):
                if not self.check_mode:
                    try:
                        self.client.post_json('/plugins/{0}/disable', self.preferred_name)
                    except APIError as e:
                        self.client.fail(to_native(e))
                self.actions.append('Disable plugin %s' % self.preferred_name)
                self.changed = True
        else:
            self.client.fail('Plugin not found: Plugin does not exist.')

    @property
    def result(self):
        plugin_data = {}
        if self.parameters.state != 'absent':
            try:
                plugin_data = self.client.get_json('/plugins/{0}/json', self.preferred_name)
            except NotFound:
                pass
        result = {'actions': self.actions, 'changed': self.changed, 'diff': self.diff, 'plugin': plugin_data}
        return dict(((k, v) for k, v in result.items() if v is not None))