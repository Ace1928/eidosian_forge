from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import (
class AnsibleCloudscaleServer(AnsibleCloudscaleBase):

    def __init__(self, module):
        super(AnsibleCloudscaleServer, self).__init__(module)
        self._info = {}

    def _init_server_container(self):
        return {'uuid': self._module.params.get('uuid') or self._info.get('uuid'), 'name': self._module.params.get('name') or self._info.get('name'), 'state': 'absent'}

    def _get_server_info(self, refresh=False):
        if self._info and (not refresh):
            return self._info
        self._info = self._init_server_container()
        uuid = self._info.get('uuid')
        if uuid is not None:
            server_info = self._get('servers/%s' % uuid)
            if server_info:
                self._info = self._transform_state(server_info)
        else:
            name = self._info.get('name')
            if name is not None:
                servers = self._get('servers') or []
                matching_server = []
                for server in servers:
                    if server['name'] == name:
                        matching_server.append(server)
                if len(matching_server) == 1:
                    self._info = self._transform_state(matching_server[0])
                elif len(matching_server) > 1:
                    self._module.fail_json(msg="More than one server with name '%s' exists. Use the 'uuid' parameter to identify the server." % name)
        return self._info

    @staticmethod
    def _transform_state(server):
        if 'status' in server:
            server['state'] = server['status']
            del server['status']
        else:
            server['state'] = 'absent'
        return server

    def _wait_for_state(self, states):
        start = datetime.now()
        timeout = self._module.params['api_timeout'] * 2
        while datetime.now() - start < timedelta(seconds=timeout):
            server_info = self._get_server_info(refresh=True)
            if server_info.get('state') in states:
                return server_info
            sleep(1)
        if server_info.get('name') is not None:
            msg = 'Timeout while waiting for a state change on server %s to states %s. Current state is %s.' % (server_info.get('name'), states, server_info.get('state'))
        else:
            name_uuid = self._module.params.get('name') or self._module.params.get('uuid')
            msg = 'Timeout while waiting to find the server %s' % name_uuid
        self._module.fail_json(msg=msg)

    def _start_stop_server(self, server_info, target_state='running', ignore_diff=False):
        actions = {'stopped': 'stop', 'running': 'start'}
        server_state = server_info.get('state')
        if server_state != target_state:
            self._result['changed'] = True
            if not ignore_diff:
                self._result['diff']['before'].update({'state': server_info.get('state')})
                self._result['diff']['after'].update({'state': target_state})
            if not self._module.check_mode:
                self._post('servers/%s/%s' % (server_info['uuid'], actions[target_state]))
                server_info = self._wait_for_state((target_state,))
        return server_info

    def _update_param(self, param_key, server_info, requires_stop=False):
        param_value = self._module.params.get(param_key)
        if param_value is None:
            return server_info
        if 'slug' in server_info[param_key]:
            server_v = server_info[param_key]['slug']
        else:
            server_v = server_info[param_key]
        if server_v != param_value:
            self._result['diff']['before'].update({param_key: server_v})
            self._result['diff']['after'].update({param_key: param_value})
            if server_info.get('state') == 'running':
                if requires_stop and (not self._module.params.get('force')):
                    self._module.warn("Some changes won't be applied to running servers. Use force=true to allow the server '%s' to be stopped/started." % server_info['name'])
                    return server_info
            self._result['changed'] = True
            if not self._module.check_mode:
                if requires_stop:
                    self._start_stop_server(server_info, target_state='stopped', ignore_diff=True)
                patch_data = {param_key: param_value}
                self._patch('servers/%s' % server_info['uuid'], patch_data)
                server_info = self._wait_for_state(('stopped', 'running'))
        return server_info

    def _get_server_group_ids(self):
        server_group_params = self._module.params['server_groups']
        if not server_group_params:
            return None
        matching_group_names = []
        results = []
        server_groups = self._get('server-groups')
        for server_group in server_groups:
            if server_group['uuid'] in server_group_params:
                results.append(server_group['uuid'])
                server_group_params.remove(server_group['uuid'])
            elif server_group['name'] in server_group_params:
                results.append(server_group['uuid'])
                server_group_params.remove(server_group['name'])
                matching_group_names.append(server_group['name'])
            elif server_group['name'] in matching_group_names:
                self._module.fail_json(msg="More than one server group with name exists: '%s'. Use the 'uuid' parameter to identify the server group." % server_group['name'])
        if server_group_params:
            self._module.fail_json(msg='Server group name or UUID not found: %s' % ', '.join(server_group_params))
        return results

    def _create_server(self, server_info):
        self._result['changed'] = True
        self.normalize_interfaces_param()
        data = deepcopy(self._module.params)
        for i in ('uuid', 'state', 'force', 'api_timeout', 'api_token', 'api_url'):
            del data[i]
        data['server_groups'] = self._get_server_group_ids()
        self._result['diff']['before'] = self._init_server_container()
        self._result['diff']['after'] = deepcopy(data)
        if not self._module.check_mode:
            self._post('servers', data)
            server_info = self._wait_for_state(('running',))
        return server_info

    def _update_server(self, server_info):
        previous_state = server_info.get('state')
        desired_server_group_ids = self._get_server_group_ids()
        if desired_server_group_ids is not None:
            current_server_group_ids = [grp['uuid'] for grp in server_info['server_groups']]
            if desired_server_group_ids != current_server_group_ids:
                self._module.warn('Server groups can not be mutated, server needs redeployment to change groups.')
        self.normalize_interfaces_param()
        wanted = self._module.params.get('interfaces')
        actual = server_info.get('interfaces')
        try:
            update_interfaces = not self.has_wanted_interfaces(wanted, actual)
        except KeyError as e:
            self._module.fail_json(msg="Error checking 'interfaces', missing key: %s" % e.args[0])
        if update_interfaces:
            server_info = self._update_param('interfaces', server_info)
            if not self._result['changed']:
                self._result['changed'] = server_info['interfaces'] != actual
        server_info = self._update_param('flavor', server_info, requires_stop=True)
        server_info = self._update_param('name', server_info)
        server_info = self._update_param('tags', server_info)
        if previous_state == 'running':
            server_info = self._start_stop_server(server_info, target_state='running', ignore_diff=True)
        return server_info

    def present_server(self):
        server_info = self._get_server_info()
        if server_info.get('state') != 'absent':
            if self._module.params.get('state') == 'stopped':
                server_info = self._start_stop_server(server_info, target_state='stopped')
            server_info = self._update_server(server_info)
            if self._module.params.get('state') == 'running':
                server_info = self._start_stop_server(server_info, target_state='running')
        else:
            server_info = self._create_server(server_info)
            server_info = self._start_stop_server(server_info, target_state=self._module.params.get('state'))
        return server_info

    def absent_server(self):
        server_info = self._get_server_info()
        if server_info.get('state') != 'absent':
            self._result['changed'] = True
            self._result['diff']['before'] = deepcopy(server_info)
            self._result['diff']['after'] = self._init_server_container()
            if not self._module.check_mode:
                self._delete('servers/%s' % server_info['uuid'])
                server_info = self._wait_for_state(('absent',))
        return server_info

    def has_wanted_interfaces(self, wanted, actual):
        """ Compares the interfaces as specified by the user, with the
        interfaces as reported by the server.

        """
        if len(wanted or ()) != len(actual or ()):
            return False

        def match_interface(spec):
            for interface in actual:
                if spec.get('network') == 'public':
                    if interface['type'] == 'public':
                        break
                if spec.get('network') is not None:
                    if interface['type'] == 'private':
                        if interface['network']['uuid'] == spec['network']:
                            break
                wanted_subnet_ids = set((a['subnet'] for a in spec.get('addresses') or ()))
                actual_subnet_ids = set((a['subnet']['uuid'] for a in interface['addresses']))
                if wanted_subnet_ids == actual_subnet_ids:
                    break
            else:
                return False
            for wanted_addr in spec.get('addresses') or ():
                if 'address' not in wanted_addr:
                    continue
                addresses = set((a['address'] for a in interface['addresses']))
                if wanted_addr['address'] not in addresses:
                    return False
            if spec.get('addresses') == [] and interface['addresses'] != []:
                return False
            if interface['addresses'] == [] and spec.get('addresses') != []:
                return False
            return interface
        for spec in wanted:
            if not match_interface(spec):
                return False
        return True

    def normalize_interfaces_param(self):
        """ Goes through the interfaces parameter and gets it ready to be
        sent to the API. """
        for spec in self._module.params.get('interfaces') or ():
            if spec['addresses'] is None:
                del spec['addresses']
            if spec['network'] is None:
                del spec['network']
            for address in spec.get('addresses') or ():
                if address['address'] is None:
                    del address['address']
                if address['subnet'] is None:
                    del address['subnet']