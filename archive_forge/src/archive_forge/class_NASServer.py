from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
class NASServer(object):
    """Class with NAS Server operations"""

    def __init__(self):
        """ Define all parameters required by this module"""
        self.module_params = utils.get_unity_management_host_parameters()
        self.module_params.update(get_nasserver_parameters())
        mut_ex_args = [['nas_server_name', 'nas_server_id']]
        required_one_of = [['nas_server_name', 'nas_server_id']]
        self.module = AnsibleModule(argument_spec=self.module_params, supports_check_mode=False, mutually_exclusive=mut_ex_args, required_one_of=required_one_of)
        utils.ensure_required_libs(self.module)
        self.result = {'changed': False, 'nas_server_details': {}}
        self.unity_conn = utils.get_unity_unisphere_connection(self.module.params, application_type)
        self.nas_server_conn_obj = utils.nas_server.UnityNasServer(self.unity_conn)
        LOG.info('Connection established with the Unity Array')

    def get_current_uds_enum(self, current_uds):
        """
        Get the enum of the Offline Availability parameter.
        :param current_uds: Current Unix Directory Service string
        :return: current_uds enum
        """
        if current_uds in utils.NasServerUnixDirectoryServiceEnum.__members__:
            return utils.NasServerUnixDirectoryServiceEnum[current_uds]
        else:
            error_msg = 'Invalid value {0} for Current Unix Directory Service provided'.format(current_uds)
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def get_nas_server(self, nas_server_name, nas_server_id):
        """
        Get the NAS Server Object using NAME/ID of the NAS Server.
        :param nas_server_name: Name of the NAS Server
        :param nas_server_id: ID of the NAS Server
        :return: NAS Server object.
        """
        nas_server = nas_server_name if nas_server_name else nas_server_id
        try:
            obj_nas = self.unity_conn.get_nas_server(_id=nas_server_id, name=nas_server_name)
            if nas_server_id and obj_nas and (not obj_nas.existed):
                LOG.error('NAS Server object does not exist with ID: %s ', nas_server_id)
                return None
            return obj_nas
        except utils.HttpError as e:
            if e.http_status == 401:
                cred_err = 'Incorrect username or password , {0}'.format(e.message)
                self.module.fail_json(msg=cred_err)
            else:
                err_msg = 'Failed to get details of NAS Server {0} with error {1}'.format(nas_server, str(e))
                LOG.error(err_msg)
                self.module.fail_json(msg=err_msg)
        except utils.UnityResourceNotFoundError as e:
            err_msg = 'Failed to get details of NAS Server {0} with error {1}'.format(nas_server, str(e))
            LOG.error(err_msg)
            return None
        except Exception as e:
            nas_server = nas_server_name if nas_server_name else nas_server_id
            err_msg = 'Failed to get nas server details {0} with error {1}'.format(nas_server, str(e))
            LOG.error(err_msg)
            self.module.fail_json(msg=err_msg)

    def to_update(self, nas_server_obj, current_uds):
        LOG.info('Checking Whether the parameters are modified or not.')
        if current_uds is not None and current_uds != nas_server_obj.current_unix_directory_service:
            return True
        if self.module.params['nas_server_new_name'] is not None and self.module.params['nas_server_new_name'] != nas_server_obj.name:
            return True
        if self.module.params['is_replication_destination'] is not None and (nas_server_obj.is_replication_destination is None or self.module.params['is_replication_destination'] != nas_server_obj.is_replication_destination):
            return True
        if self.module.params['is_multiprotocol_enabled'] is not None and (nas_server_obj.is_multi_protocol_enabled is None or self.module.params['is_multiprotocol_enabled'] != nas_server_obj.is_multi_protocol_enabled):
            return True
        if self.module.params['is_backup_only'] is not None and (nas_server_obj.is_backup_only is None or self.module.params['is_backup_only'] != nas_server_obj.is_backup_only):
            return True
        if self.module.params['is_packet_reflect_enabled'] is not None and (nas_server_obj.is_packet_reflect_enabled is None or self.module.params['is_packet_reflect_enabled'] != nas_server_obj.is_packet_reflect_enabled):
            return True
        if self.module.params['allow_unmapped_user'] is not None and (nas_server_obj.allow_unmapped_user is None or self.module.params['allow_unmapped_user'] != nas_server_obj.allow_unmapped_user):
            return True
        nas_win_flag = nas_server_obj.is_windows_to_unix_username_mapping_enabled
        input_win_flag = self.module.params['enable_windows_to_unix_username_mapping']
        if input_win_flag is not None and (nas_win_flag is None or nas_win_flag != input_win_flag):
            return True
        if self.module.params['default_windows_user'] is not None and (nas_server_obj.default_windows_user is None or self.module.params['default_windows_user'] != nas_server_obj.default_windows_user):
            return True
        if self.module.params['default_unix_user'] is not None and (nas_server_obj.default_unix_user is None or self.module.params['default_unix_user'] != nas_server_obj.default_unix_user):
            return True
        return False

    def update_nas_server(self, nas_server_obj, new_name=None, default_unix_user=None, default_windows_user=None, is_rep_dest=None, is_multiprotocol_enabled=None, allow_unmapped_user=None, is_backup_only=None, is_packet_reflect_enabled=None, current_uds=None, enable_win_to_unix_name_map=None):
        """
        The Details of the NAS Server will be updated in the function.
        """
        try:
            nas_server_obj.modify(name=new_name, is_replication_destination=is_rep_dest, is_backup_only=is_backup_only, is_multi_protocol_enabled=is_multiprotocol_enabled, default_unix_user=default_unix_user, default_windows_user=default_windows_user, allow_unmapped_user=allow_unmapped_user, is_packet_reflect_enabled=is_packet_reflect_enabled, enable_windows_to_unix_username=enable_win_to_unix_name_map, current_unix_directory_service=current_uds)
        except Exception as e:
            error_msg = 'Failed to Update parameters of NAS Server %s with error %s' % (nas_server_obj.name, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def modify_replication_session(self, nas_server_obj, repl_session, replication_params):
        """ Modify the replication session
            :param: nas_server_obj: NAS server object
            :param: repl_session: Replication session to be modified
            :param: replication_params: Module input params
            :return: True if modification is successful
        """
        try:
            LOG.info('Modifying replication session of nas server %s', nas_server_obj.name)
            modify_payload = {}
            if replication_params['replication_mode'] and replication_params['replication_mode'] == 'manual':
                rpo = -1
            elif replication_params['rpo']:
                rpo = replication_params['rpo']
            name = repl_session.name
            if replication_params['new_replication_name'] and name != replication_params['new_replication_name']:
                name = replication_params['new_replication_name']
            if repl_session.name != name:
                modify_payload['name'] = name
            if (replication_params['replication_mode'] or replication_params['rpo']) and repl_session.max_time_out_of_sync != rpo:
                modify_payload['max_time_out_of_sync'] = rpo
            if modify_payload:
                repl_session.modify(**modify_payload)
                return True
            return False
        except Exception as e:
            errormsg = ('Modifying replication session failed with error %s', e)
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def enable_replication(self, nas_server_obj, replication, replication_reuse_resource):
        """ Enable replication on NAS Server
            :param: nas_server_obj: NAS Server object.
            :param: replication: Dict which has all the replication parameter values.
            :return: True if replication is enabled else False.
        """
        try:
            self.validate_nas_server_replication_params(replication)
            self.update_replication_params(replication, replication_reuse_resource)
            repl_session = self.get_replication_session_on_filter(nas_server_obj, replication, 'modify')
            if repl_session:
                return self.modify_replication_session(nas_server_obj, repl_session, replication)
            self.validate_create_replication_params(replication)
            replication_args_list = get_replication_args_list(replication)
            if 'replication_type' in replication and replication['replication_type'] == 'remote':
                self.get_remote_system(replication, replication_args_list)
                if not replication_reuse_resource:
                    update_replication_arg_list(replication, replication_args_list, nas_server_obj)
                    nas_server_obj.replicate_with_dst_resource_provisioning(**replication_args_list)
                else:
                    replication_args_list['dst_nas_server_id'] = replication['destination_nas_server_id']
                    nas_server_obj.replicate(**replication_args_list)
                return True
            if 'replication_type' in replication and replication['replication_type'] == 'local':
                update_replication_arg_list(replication, replication_args_list, nas_server_obj)
                nas_server_obj.replicate_with_dst_resource_provisioning(**replication_args_list)
                return True
        except Exception as e:
            errormsg = 'Enabling replication to the nas server %s failed with error %s' % (nas_server_obj.name, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def disable_replication(self, obj_nas, replication_params):
        """ Remove replication from the nas server
            :param: replication_params: Module input params
            :param: obj_nas: NAS Server object
            :return: True if disabling replication is successful
        """
        try:
            LOG.info(('Disabling replication on the nas server %s', obj_nas.name))
            if replication_params:
                self.update_replication_params(replication_params, False)
            repl_session = self.get_replication_session_on_filter(obj_nas, replication_params, 'delete')
            if repl_session:
                repl_session.delete()
                return True
            return False
        except Exception as e:
            errormsg = 'Disabling replication on the nas server %s failed with error %s' % (obj_nas.name, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def get_replication_session_on_filter(self, obj_nas, replication_params, action):
        """ Retrieves replication session on nas server
            :param: obj_nas: NAS server object
            :param: replication_params: Module input params
            :param: action: Specifies action as modify or delete
            :return: Replication session based on filter
        """
        if replication_params and replication_params['remote_system']:
            repl_session = self.get_replication_session(obj_nas, filter_key='remote_system_name', replication_params=replication_params)
        elif replication_params and replication_params['replication_name']:
            repl_session = self.get_replication_session(obj_nas, filter_key='name', name=replication_params['replication_name'])
        else:
            repl_session = self.get_replication_session(obj_nas, action=action)
            if repl_session and action and replication_params and (replication_params['replication_type'] == 'local') and (repl_session.remote_system.name != self.unity_conn.name):
                return None
        return repl_session

    def get_replication_session(self, obj_nas, filter_key=None, replication_params=None, name=None, action=None):
        """ Retrieves the replication sessions configured for the nas server
            :param: obj_nas: NAS server object
            :param: filter_key: Key to filter replication sessions
            :param: replication_params: Module input params
            :param: name: Replication session name
            :param: action: Specifies modify or delete action on replication session
            :return: Replication session details
        """
        try:
            repl_session = self.unity_conn.get_replication_session(src_resource_id=obj_nas.id)
            if not filter_key and repl_session:
                if len(repl_session) > 1:
                    if action:
                        error_msg = 'There are multiple replication sessions for the nas server. Please specify replication_name in replication_params to %s.' % action
                        self.module.fail_json(msg=error_msg)
                    return repl_session
                return repl_session[0]
            for session in repl_session:
                if filter_key == 'remote_system_name' and session.remote_system.name == replication_params['remote_system_name']:
                    return session
                if filter_key == 'name' and session.name == name:
                    return session
            return None
        except Exception as e:
            errormsg = ('Retrieving replication session on the nas server failed with error %s', str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def get_remote_system(self, replication, replication_args_list):
        remote_system_name = replication['remote_system_name']
        remote_system_list = self.unity_conn.get_remote_system()
        for remote_system in remote_system_list:
            if remote_system.name == remote_system_name:
                replication_args_list['remote_system'] = remote_system
                break
        if 'remote_system' not in replication_args_list.keys():
            errormsg = 'Remote system %s is not found' % remote_system_name
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def update_replication_params(self, replication, replication_reuse_resource):
        """ Update replication dict with remote system information
            :param: replication: Dict which has all the replication parameter values
            :return: Updated replication Dict
        """
        try:
            if 'replication_type' in replication and replication['replication_type'] == 'remote':
                connection_params = {'unispherehost': replication['remote_system']['remote_system_host'], 'username': replication['remote_system']['remote_system_username'], 'password': replication['remote_system']['remote_system_password'], 'validate_certs': replication['remote_system']['remote_system_verifycert'], 'port': replication['remote_system']['remote_system_port']}
                remote_system_conn = utils.get_unity_unisphere_connection(connection_params, application_type)
                replication['remote_system_name'] = remote_system_conn.name
                if replication['destination_pool_name'] is not None:
                    pool_object = remote_system_conn.get_pool(name=replication['destination_pool_name'])
                    replication['destination_pool_id'] = pool_object.id
                if replication['destination_nas_server_name'] is not None and replication_reuse_resource:
                    nas_object = remote_system_conn.get_nas_server(name=replication['destination_nas_server_name'])
                    replication['destination_nas_server_id'] = nas_object.id
            else:
                replication['remote_system_name'] = self.unity_conn.name
                if replication['destination_pool_name'] is not None:
                    pool_object = self.unity_conn.get_pool(name=replication['destination_pool_name'])
                    replication['destination_pool_id'] = pool_object.id
        except Exception as e:
            errormsg = 'Updating replication params failed with error %s' % str(e)
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def validate_rpo(self, replication):
        if 'replication_mode' in replication and replication['replication_mode'] == 'asynchronous' and (replication['rpo'] is None):
            errormsg = "rpo is required together with 'asynchronous' replication_mode."
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        if (replication['rpo'] and (replication['rpo'] < 5 or replication['rpo'] > 1440)) and (replication['replication_mode'] and replication['replication_mode'] != 'manual' or (not replication['replication_mode'] and replication['rpo'] != -1)):
            errormsg = 'rpo value should be in range of 5 to 1440'
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def validate_nas_server_replication_params(self, replication):
        """ Validate NAS server replication params
            :param: replication: Dict which has all the replication parameter values
        """
        if replication is None:
            errormsg = 'Please specify replication_params to enable replication.'
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        else:
            if replication['destination_pool_id'] is not None and replication['destination_pool_name'] is not None:
                errormsg = "'destination_pool_id' and 'destination_pool_name' is mutually exclusive."
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)
            self.validate_rpo(replication)
            if replication['replication_type'] == 'remote' and replication['remote_system'] is None:
                errormsg = "Remote_system is required together with 'remote' replication_type"
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)
            if 'destination_nas_name' in replication and replication['destination_nas_name'] is not None:
                dst_nas_server_name_length = len(replication['destination_nas_name'])
                if dst_nas_server_name_length == 0 or dst_nas_server_name_length > 95:
                    errormsg = 'destination_nas_name value should be in range of 1 to 95'
                    LOG.error(errormsg)
                    self.module.fail_json(msg=errormsg)

    def validate_create_replication_params(self, replication):
        """ Validate replication params """
        if replication['destination_pool_id'] is None and replication['destination_pool_name'] is None:
            errormsg = "Either 'destination_pool_id' or 'destination_pool_name' is required."
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        keys = ['replication_mode', 'replication_type']
        for key in keys:
            if replication[key] is None:
                errormsg = 'Please specify %s to enable replication.' % key
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)

    def perform_module_operation(self):
        """
        Perform different actions on NAS Server based on user parameters
        chosen in playbook
        """
        state = self.module.params['state']
        nas_server_name = self.module.params['nas_server_name']
        nas_server_id = self.module.params['nas_server_id']
        nas_server_new_name = self.module.params['nas_server_new_name']
        default_unix_user = self.module.params['default_unix_user']
        default_windows_user = self.module.params['default_windows_user']
        is_replication_destination = self.module.params['is_replication_destination']
        is_multiprotocol_enabled = self.module.params['is_multiprotocol_enabled']
        allow_unmapped_user = self.module.params['allow_unmapped_user']
        enable_windows_to_unix_username_mapping = self.module.params['enable_windows_to_unix_username_mapping']
        is_backup_only = self.module.params['is_backup_only']
        is_packet_reflect_enabled = self.module.params['is_packet_reflect_enabled']
        current_uds = self.module.params['current_unix_directory_service']
        replication = self.module.params['replication_params']
        replication_state = self.module.params['replication_state']
        replication_reuse_resource = self.module.params['replication_reuse_resource']
        if current_uds:
            current_uds = self.get_current_uds_enum(current_uds)
        changed = False
        if replication and replication_state is None:
            self.module.fail_json(msg='Please specify replication_state along with replication_params')
        '\n        Get details of NAS Server.\n        '
        nas_server_obj = None
        if nas_server_name or nas_server_id:
            nas_server_obj = self.get_nas_server(nas_server_name, nas_server_id)
        if not nas_server_obj and state == 'present':
            msg = 'NAS Server Resource not found. Please enter a valid Name/ID to get or modify the parameters of nas server.'
            LOG.error(msg)
            self.module.fail_json(msg=msg)
        '\n            Update the parameters of NAS Server\n        '
        if nas_server_obj and state == 'present':
            update_flag = self.to_update(nas_server_obj, current_uds)
            if update_flag:
                self.update_nas_server(nas_server_obj, nas_server_new_name, default_unix_user, default_windows_user, is_replication_destination, is_multiprotocol_enabled, allow_unmapped_user, is_backup_only, is_packet_reflect_enabled, current_uds, enable_windows_to_unix_username_mapping)
                changed = True
        if nas_server_obj and state == 'absent':
            self.module.fail_json(msg='Deletion of NAS Server is currently not supported.')
        if state == 'present' and nas_server_obj and (replication_state is not None):
            if replication_state == 'enable':
                changed = self.enable_replication(nas_server_obj, replication, replication_reuse_resource)
            else:
                changed = self.disable_replication(nas_server_obj, replication)
        '\n            Update the changed state and NAS Server details\n        '
        nas_server_details = None
        if nas_server_obj:
            nas_server_details = self.get_nas_server(None, nas_server_obj.id)._get_properties()
        self.result['changed'] = changed
        self.result['nas_server_details'] = nas_server_details
        self.module.exit_json(**self.result)