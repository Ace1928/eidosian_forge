from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
class NFSServer(object):
    """Class with NFS server operations"""

    def __init__(self):
        """Define all parameters required by this module"""
        self.module_params = utils.get_unity_management_host_parameters()
        self.module_params.update(get_nfs_server_parameters())
        mutually_exclusive = [['nas_server_name', 'nas_server_id']]
        required_one_of = [['nfs_server_id', 'nas_server_name', 'nas_server_id']]
        self.module = AnsibleModule(argument_spec=self.module_params, supports_check_mode=True, mutually_exclusive=mutually_exclusive, required_one_of=required_one_of)
        utils.ensure_required_libs(self.module)
        self.unity_conn = utils.get_unity_unisphere_connection(self.module.params, application_type)
        LOG.info('Check Mode Flag %s', self.module.check_mode)

    def get_nfs_server_details(self, nfs_server_id=None, nas_server_id=None):
        """Get NFS server details.
            :param: nfs_server_id: The ID of the NFS server
            :param: nas_server_id: The name of the NAS server
            :return: Dict containing NFS server details if exists
        """
        LOG.info('Getting NFS server details')
        try:
            if nfs_server_id:
                nfs_server_details = self.unity_conn.get_nfs_server(_id=nfs_server_id)
                return nfs_server_details._get_properties()
            elif nas_server_id:
                nfs_server_details = self.unity_conn.get_nfs_server(nas_server=nas_server_id)
                if len(nfs_server_details) > 0:
                    return process_dict(nfs_server_details._get_properties())
                return None
        except utils.HttpError as e:
            if e.http_status == 401:
                msg = 'Incorrect username or password provided.'
                LOG.error(msg)
                self.module.fail_json(msg=msg)
            else:
                err_msg = 'Failed to get details of NFS Server with error {0}'.format(str(e))
                LOG.error(err_msg)
                self.module.fail_json(msg=err_msg)
        except utils.UnityResourceNotFoundError as e:
            err_msg = 'Failed to get details of NFS Server with error {0}'.format(str(e))
            LOG.error(err_msg)
            return None

    def get_nfs_server_instance(self, nfs_server_id):
        """Get NFS server instance.
            :param: nfs_server_id: The ID of the NFS server
            :return: Return NFS server instance if exists
        """
        try:
            nfs_server_obj = self.unity_conn.get_nfs_server(_id=nfs_server_id)
            return nfs_server_obj
        except Exception as e:
            error_msg = 'Failed to get the NFS server %s instance with error %s' % (nfs_server_id, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def delete_nfs_server(self, nfs_server_id, skip_unjoin=None, domain_username=None, domain_password=None):
        """Delete NFS server.
            :param: nfs_server_id: The ID of the NFS server
            :param: skip_unjoin: Flag indicating whether to unjoin SMB server account from AD before deletion
            :param: domain_username: The domain username
            :param: domain_password: The domain password
            :return: Return True if NFS server is deleted
        """
        LOG.info('Deleting NFS server')
        try:
            if not self.module.check_mode:
                nfs_obj = self.get_nfs_server_instance(nfs_server_id=nfs_server_id)
                nfs_obj.delete(skip_kdc_unjoin=skip_unjoin, username=domain_username, password=domain_password)
            return True
        except Exception as e:
            msg = 'Failed to delete NFS server: %s with error: %s' % (nfs_server_id, str(e))
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def get_nas_server_id(self, nas_server_name):
        """Get NAS server ID.
            :param: nas_server_name: The name of NAS server
            :return: Return NAS server ID if exists
        """
        LOG.info('Getting NAS server ID')
        try:
            obj_nas = self.unity_conn.get_nas_server(name=nas_server_name)
            return obj_nas.get_id()
        except Exception as e:
            msg = 'Failed to get details of NAS server: %s with error: %s' % (nas_server_name, str(e))
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def is_modification_required(self, is_extended_credentials_enabled, nfs_server_details):
        """Check if modification is required in existing NFS server
            :param: is_extended_credentials_enabled: Indicates whether the NFS server supports more than 16 Unix groups in a Unix credential.
            :param: nfs_server_details: NFS server details
            :return: True if modification is required
        """
        LOG.info('Checking if any modification is required')
        if is_extended_credentials_enabled is not None and is_extended_credentials_enabled != nfs_server_details['is_extended_credentials_enabled']:
            return True

    def create_nfs_server(self, nas_server_id, host_name=None, nfs_v4_enabled=None, is_secure_enabled=None, kerberos_domain_controller_type=None, kerberos_domain_controller_username=None, kerberos_domain_controller_password=None, is_extended_credentials_enabled=None):
        """Create NFS server.
            :param: nas_server_id: The ID of NAS server.
            :param: host_name: Name of NFS Server.
            :param: nfs_v4_enabled: Indicates whether the NFSv4 is enabled on the NAS server.
            :param: is_secure_enabled: Indicates whether the secure NFS is enabled.
            :param: kerberos_domain_controller_type: Type of Kerberos Domain Controller used for secure NFS service.
            :param: kerberos_domain_controller_username: Kerberos Domain Controller administrator username.
            :param: kerberos_domain_controller_password: Kerberos Domain Controller administrator password.
            :param: is_extended_credentials_enabled: Indicates whether support for more than 16 unix groups in a Unix credential.
        """
        LOG.info('Creating NFS server')
        try:
            if not self.module.check_mode:
                kdc_enum_type = get_enum_kdctype(kerberos_domain_controller_type)
                if kerberos_domain_controller_type == 'UNIX':
                    is_extended_credentials_enabled = None
                    is_secure_enabled = None
                utils.UnityNfsServer.create(cli=self.unity_conn._cli, nas_server=nas_server_id, host_name=host_name, nfs_v4_enabled=nfs_v4_enabled, is_secure_enabled=is_secure_enabled, kdc_type=kdc_enum_type, kdc_username=kerberos_domain_controller_username, kdc_password=kerberos_domain_controller_password, is_extended_credentials_enabled=is_extended_credentials_enabled)
            return True
        except Exception as e:
            msg = 'Failed to create NFS server with on NAS Server %s with error: %s' % (nas_server_id, str(e))
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def validate_input_params(self):
        param_list = ['nfs_server_id', 'nas_server_id', 'nas_server_name', 'host_name', 'kerberos_domain_controller_username', 'kerberos_domain_controller_password']
        for param in param_list:
            msg = 'Please provide valid value for: %s' % param
            if self.module.params[param] is not None and len(self.module.params[param].strip()) == 0:
                errmsg = msg.format(param)
                self.module.fail_json(msg=errmsg)

    def perform_module_operation(self):
        """
        Perform different actions on NFS server module based on parameters
        passed in the playbook
        """
        nfs_server_id = self.module.params['nfs_server_id']
        nas_server_id = self.module.params['nas_server_id']
        nas_server_name = self.module.params['nas_server_name']
        host_name = self.module.params['host_name']
        nfs_v4_enabled = self.module.params['nfs_v4_enabled']
        is_secure_enabled = self.module.params['is_secure_enabled']
        kerberos_domain_controller_type = self.module.params['kerberos_domain_controller_type']
        kerberos_domain_controller_username = self.module.params['kerberos_domain_controller_username']
        kerberos_domain_controller_password = self.module.params['kerberos_domain_controller_password']
        is_extended_credentials_enabled = self.module.params['is_extended_credentials_enabled']
        remove_spn_from_kerberos = self.module.params['remove_spn_from_kerberos']
        state = self.module.params['state']
        result = dict(changed=False, nfs_server_details={})
        modify_flag = False
        self.validate_input_params()
        if nas_server_name:
            nas_server_id = self.get_nas_server_id(nas_server_name)
        nfs_server_details = self.get_nfs_server_details(nfs_server_id=nfs_server_id, nas_server_id=nas_server_id)
        if nfs_server_details and state == 'present':
            modify_flag = self.is_modification_required(is_extended_credentials_enabled, nfs_server_details)
            if modify_flag:
                self.module.fail_json(msg='Modification of NFS Server parameters is not supported through Ansible module')
        if not nfs_server_details and state == 'present':
            if not nas_server_id:
                self.module.fail_json(msg='Please provide nas server id/name to create NFS server.')
            result['changed'] = self.create_nfs_server(nas_server_id, host_name, nfs_v4_enabled, is_secure_enabled, kerberos_domain_controller_type, kerberos_domain_controller_username, kerberos_domain_controller_password, is_extended_credentials_enabled)
        if state == 'absent' and nfs_server_details:
            skip_unjoin = not remove_spn_from_kerberos
            result['changed'] = self.delete_nfs_server(nfs_server_details['id'], skip_unjoin, kerberos_domain_controller_username, kerberos_domain_controller_password)
        if state == 'present':
            result['nfs_server_details'] = self.get_nfs_server_details(nfs_server_id=nfs_server_id, nas_server_id=nas_server_id)
        self.module.exit_json(**result)