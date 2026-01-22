from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
class PowerFlexSDS(PowerFlexBase):
    """Class with SDS operations"""

    def __init__(self):
        """ Define all parameters required by this module"""
        self.module_params = utils.get_powerflex_gateway_host_parameters()
        self.module_params.update(get_powerflex_sds_parameters())
        mut_ex_args = [['sds_name', 'sds_id'], ['protection_domain_name', 'protection_domain_id'], ['fault_set_name', 'fault_set_id']]
        required_together_args = [['sds_ip_list', 'sds_ip_state']]
        required_one_of_args = [['sds_name', 'sds_id']]
        ansible_module_params = {'argument_spec': get_powerflex_sds_parameters(), 'supports_check_mode': True, 'mutually_exclusive': mut_ex_args, 'required_one_of': required_one_of_args, 'required_together': required_together_args}
        super().__init__(AnsibleModule, ansible_module_params)
        self.result = dict(changed=False, sds_details={})

    def validate_rmcache_size_parameter(self, rmcache_enabled, rmcache_size):
        """Validate the input parameters"""
        if rmcache_size is not None and rmcache_enabled is False:
            error_msg = 'RM cache size can be set only when RM cache is enabled, please enable it along with RM cache size.'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def validate_ip_parameter(self, sds_ip_list):
        """Validate the input parameters"""
        if sds_ip_list is None or len(sds_ip_list) == 0:
            error_msg = "Provide valid values for sds_ip_list as 'ip' and 'role' for Create/Modify operations."
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def get_sds_details(self, sds_name=None, sds_id=None):
        """Get SDS details
            :param sds_name: Name of the SDS
            :type sds_name: str
            :param sds_id: ID of the SDS
            :type sds_id: str
            :return: Details of SDS if it exist
            :rtype: dict
        """
        id_or_name = sds_id if sds_id else sds_name
        try:
            if sds_name:
                sds_details = self.powerflex_conn.sds.get(filter_fields={'name': sds_name})
            else:
                sds_details = self.powerflex_conn.sds.get(filter_fields={'id': sds_id})
            if len(sds_details) == 0:
                msg = "SDS with identifier '%s' not found" % id_or_name
                LOG.info(msg)
                return None
            return sds_details[0]
        except Exception as e:
            error_msg = "Failed to get the SDS '%s' with error '%s'" % (id_or_name, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def get_protection_domain(self, protection_domain_name=None, protection_domain_id=None):
        """Get the details of a protection domain in a given PowerFlex storage
        system"""
        return Configuration(self.powerflex_conn, self.module).get_protection_domain(protection_domain_name=protection_domain_name, protection_domain_id=protection_domain_id)

    def get_fault_set(self, fault_set_name=None, fault_set_id=None, protection_domain_id=None):
        """Get fault set details
            :param fault_set_name: Name of the fault set
            :param fault_set_id: Id of the fault set
            :param protection_domain_id: ID of the protection domain
            :return: Fault set details
            :rtype: dict
        """
        return Configuration(self.powerflex_conn, self.module).get_fault_set(fault_set_name=fault_set_name, fault_set_id=fault_set_id, protection_domain_id=protection_domain_id)

    def restructure_ip_role_dict(self, sds_ip_list):
        """Restructure IP role dict
            :param sds_ip_list: List of one or more IP addresses and
                                their roles
            :type sds_ip_list: list[dict]
            :return: List of one or more IP addresses and their roles
            :rtype: list[dict]
        """
        new_sds_ip_list = []
        for item in sds_ip_list:
            new_sds_ip_list.append({'SdsIp': item})
        return new_sds_ip_list

    def validate_create(self, protection_domain_id, sds_ip_list, sds_ip_state, sds_name, sds_id, sds_new_name, rmcache_enabled=None, rmcache_size=None, fault_set_id=None):
        if sds_name is None or len(sds_name.strip()) == 0:
            error_msg = 'Please provide valid sds_name value for creation of SDS.'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)
        if protection_domain_id is None:
            error_msg = 'Protection Domain is a mandatory parameter for creating an SDS. Please enter a valid value.'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)
        if sds_ip_list is None or len(sds_ip_list) == 0:
            error_msg = 'Please provide valid sds_ip_list values for creation of SDS.'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)
        if sds_ip_state is not None and sds_ip_state != 'present-in-sds':
            error_msg = 'Incorrect IP state given for creation of SDS.'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)
        if sds_id:
            error_msg = 'Creation of SDS is allowed using sds_name only, sds_id given.'
            LOG.info(error_msg)
            self.module.fail_json(msg=error_msg)

    def create_sds(self, protection_domain_id, sds_ip_list, sds_ip_state, sds_name, sds_id, sds_new_name, rmcache_enabled=None, rmcache_size=None, fault_set_id=None):
        """Create SDS
            :param protection_domain_id: ID of the Protection Domain
            :type protection_domain_id: str
            :param sds_ip_list: List of one or more IP addresses associated
                                with the SDS over which the data will be
                                transferred.
            :type sds_ip_list: list[dict]
            :param sds_ip_state: SDS IP state
            :type sds_ip_state: str
            :param sds_name: SDS name
            :type sds_name: str
            :param rmcache_enabled: Whether to enable the Read RAM cache
            :type rmcache_enabled: bool
            :param rmcache_size: Read RAM cache size (in MB)
            :type rmcache_size: int
            :param fault_set_id: ID of the Fault Set
            :type fault_set_id: str
            :return: Boolean indicating if create operation is successful
        """
        try:
            self.validate_create(protection_domain_id=protection_domain_id, sds_ip_list=sds_ip_list, sds_ip_state=sds_ip_state, sds_name=sds_name, sds_id=sds_id, sds_new_name=sds_new_name, rmcache_enabled=rmcache_enabled, rmcache_size=rmcache_size, fault_set_id=fault_set_id)
            self.validate_ip_parameter(sds_ip_list)
            if not self.module.check_mode:
                if sds_ip_list and sds_ip_state == 'present-in-sds':
                    sds_ip_list = self.restructure_ip_role_dict(sds_ip_list)
                if rmcache_size is not None:
                    self.validate_rmcache_size_parameter(rmcache_enabled=rmcache_enabled, rmcache_size=rmcache_size)
                    rmcache_size = rmcache_size * 1024
                create_params = 'protection_domain_id: %s, sds_ip_list: %s, sds_name: %s, rmcache_enabled: %s,  rmcache_size_KB: %s,  fault_set_id: %s' % (protection_domain_id, sds_ip_list, sds_name, rmcache_enabled, rmcache_size, fault_set_id)
                LOG.info('Creating SDS with params: %s', create_params)
                self.powerflex_conn.sds.create(protection_domain_id=protection_domain_id, sds_ips=sds_ip_list, name=sds_name, rmcache_enabled=rmcache_enabled, rmcache_size_in_kb=rmcache_size, fault_set_id=fault_set_id)
            return self.get_sds_details(sds_name=sds_name)
        except Exception as e:
            error_msg = f'Create SDS {sds_name} operation failed with error {str(e)}'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def to_modify(self, sds_details, sds_new_name, rfcache_enabled, rmcache_enabled, rmcache_size, performance_profile):
        """
        :param sds_details: Details of the SDS
        :type sds_details: dict
        :param sds_new_name: New name of SDS
        :type sds_new_name: str
        :param rfcache_enabled: Whether to enable the Read Flash cache
        :type rfcache_enabled: bool
        :param rmcache_enabled: Whether to enable the Read RAM cache
        :type rmcache_enabled: bool
        :param rmcache_size: Read RAM cache size (in MB)
        :type rmcache_size: int
        :param performance_profile: Performance profile to apply to the SDS
        :type performance_profile: str
        :return: Dictionary containing the attributes of SDS which are to be
                 updated
        :rtype: dict
        """
        modify_dict = {}
        if sds_new_name is not None and sds_new_name != sds_details['name']:
            modify_dict['name'] = sds_new_name
        param_input = dict()
        param_input['rfcacheEnabled'] = rfcache_enabled
        param_input['rmcacheEnabled'] = rmcache_enabled
        param_input['perfProfile'] = performance_profile
        param_list = ['rfcacheEnabled', 'rmcacheEnabled', 'perfProfile']
        for param in param_list:
            if param_input[param] is not None and sds_details[param] != param_input[param]:
                modify_dict[param] = param_input[param]
        if rmcache_size is not None:
            self.validate_rmcache_size_parameter(rmcache_enabled, rmcache_size)
            exisitng_size_mb = sds_details['rmcacheSizeInKb'] / 1024
            if rmcache_size != exisitng_size_mb:
                if sds_details['rmcacheEnabled']:
                    modify_dict['rmcacheSizeInMB'] = rmcache_size
                else:
                    error_msg = "Failed to update RM cache size for the SDS '%s' as RM cache is disabled previously, please enable it before setting the size." % sds_details['name']
                    LOG.error(error_msg)
                    self.module.fail_json(msg=error_msg)
        return modify_dict

    def modify_sds_attributes(self, sds_id, modify_dict, create_flag=False):
        """Modify SDS attributes
            :param sds_id: SDS ID
            :type sds_id: str
            :param modify_dict: Dictionary containing the attributes of SDS
                                which are to be updated
            :type modify_dict: dict
            :param create_flag: Flag to indicate whether modify operation is
                                followed by create operation or not
            :type create_flag: bool
            :return: Boolean indicating if the operation is successful
        """
        try:
            msg = "Dictionary containing attributes which are to be updated is '%s'." % str(modify_dict)
            LOG.info(msg)
            if not self.module.check_mode:
                if 'name' in modify_dict:
                    self.powerflex_conn.sds.rename(sds_id, modify_dict['name'])
                    msg = "The name of the SDS is updated to '%s' successfully." % modify_dict['name']
                    LOG.info(msg)
                if 'rfcacheEnabled' in modify_dict:
                    self.powerflex_conn.sds.set_rfcache_enabled(sds_id, modify_dict['rfcacheEnabled'])
                    msg = "The use RFcache is updated to '%s' successfully." % modify_dict['rfcacheEnabled']
                    LOG.info(msg)
                if 'rmcacheEnabled' in modify_dict:
                    self.powerflex_conn.sds.set_rmcache_enabled(sds_id, modify_dict['rmcacheEnabled'])
                    msg = "The use RMcache is updated to '%s' successfully." % modify_dict['rmcacheEnabled']
                    LOG.info(msg)
                if 'rmcacheSizeInMB' in modify_dict:
                    self.powerflex_conn.sds.set_rmcache_size(sds_id, modify_dict['rmcacheSizeInMB'])
                    msg = "The size of RMcache is updated to '%s' successfully." % modify_dict['rmcacheSizeInMB']
                    LOG.info(msg)
                if 'perfProfile' in modify_dict:
                    self.powerflex_conn.sds.set_performance_parameters(sds_id, modify_dict['perfProfile'])
                    msg = "The performance profile is updated to '%s'" % modify_dict['perfProfile']
                    LOG.info(msg)
            return self.get_sds_details(sds_id=sds_id)
        except Exception as e:
            if create_flag:
                error_msg = "Create SDS is successful, but failed to update the SDS '%s' with error '%s'" % (sds_id, str(e))
            else:
                error_msg = "Failed to update the SDS '%s' with error '%s'" % (sds_id, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def identify_ip_role_add(self, sds_ip_list, sds_details, sds_ip_state):
        existing_ip_role_list = sds_details['ipList']
        update_role = []
        ips_to_add = []
        existing_ip_list = []
        if existing_ip_role_list:
            for ip in existing_ip_role_list:
                existing_ip_list.append(ip['ip'])
        for given_ip in sds_ip_list:
            ip = given_ip['ip']
            if ip not in existing_ip_list:
                ips_to_add.append(given_ip)
        LOG.info('IP(s) to be added: %s', ips_to_add)
        if len(ips_to_add) != 0:
            for ip in ips_to_add:
                sds_ip_list.remove(ip)
        update_role = [ip for ip in sds_ip_list if ip not in existing_ip_role_list]
        LOG.info('Role update needed for: %s', update_role)
        return (ips_to_add, update_role)

    def identify_ip_role_remove(self, sds_ip_list, sds_details, sds_ip_state):
        existing_ip_role_list = sds_details['ipList']
        if sds_ip_state == 'absent-in-sds':
            ips_to_remove = [ip for ip in existing_ip_role_list if ip in sds_ip_list]
            if len(ips_to_remove) != 0:
                LOG.info('IP(s) to remove: %s', ips_to_remove)
                return ips_to_remove
            else:
                LOG.info('IP(s) do not exists.')
                return []

    def add_ip(self, sds_id, sds_ip_list):
        """Add IP to SDS
            :param sds_id: SDS ID
            :type sds_id: str
            :param sds_ip_list: List of one or more IP addresses and
                                their roles
            :type sds_ip_list: list[dict]
            :return: Boolean indicating if add IP operation is successful
        """
        try:
            if not self.module.check_mode:
                for ip in sds_ip_list:
                    LOG.info('IP to add: %s', ip)
                    self.powerflex_conn.sds.add_ip(sds_id=sds_id, sds_ip=ip)
                    LOG.info('IP added successfully.')
            return True
        except Exception as e:
            error_msg = "Add IP to SDS '%s' operation failed with error '%s'" % (sds_id, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def update_role(self, sds_id, sds_ip_list):
        """Update IP's role for an SDS
            :param sds_id: SDS ID
            :type sds_id: str
            :param sds_ip_list: List of one or more IP addresses and
                                their roles
            :type sds_ip_list: list[dict]
            :return: Boolean indicating if add IP operation is successful
        """
        try:
            if not self.module.check_mode:
                LOG.info('Role updates for: %s', sds_ip_list)
                if len(sds_ip_list) != 0:
                    for ip in sds_ip_list:
                        LOG.info('ip-role: %s', ip)
                        self.powerflex_conn.sds.set_ip_role(sds_id, ip['ip'], ip['role'])
                        msg = "The role '%s' for IP '%s' is updated successfully." % (ip['role'], ip['ip'])
                        LOG.info(msg)
            return True
        except Exception as e:
            error_msg = "Update role of IP for SDS '%s' operation failed with error '%s'" % (sds_id, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def remove_ip(self, sds_id, sds_ip_list):
        """Remove IP from SDS
            :param sds_id: SDS ID
            :type sds_id: str
            :param sds_ip_list: List of one or more IP addresses and
                                their roles.
            :type sds_ip_list: list[dict]
            :return: Boolean indicating if remove IP operation is successful
        """
        try:
            if not self.module.check_mode:
                for ip in sds_ip_list:
                    LOG.info('IP to remove: %s', ip)
                    self.powerflex_conn.sds.remove_ip(sds_id=sds_id, ip=ip['ip'])
                    LOG.info('IP removed successfully.')
            return True
        except Exception as e:
            error_msg = "Remove IP from SDS '%s' operation failed with error '%s'" % (sds_id, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def delete_sds(self, sds_id):
        """Delete SDS
            :param sds_id: SDS ID
            :type sds_id: str
            :return: Boolean indicating if delete operation is successful
        """
        try:
            if not self.module.check_mode:
                self.powerflex_conn.sds.delete(sds_id)
                return None
            return self.get_sds_details(sds_id=sds_id)
        except Exception as e:
            error_msg = "Delete SDS '%s' operation failed with error '%s'" % (sds_id, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def show_output(self, sds_id):
        """Show SDS details
            :param sds_id: ID of the SDS
            :type sds_id: str
            :return: Details of SDS
            :rtype: dict
        """
        try:
            sds_details = self.powerflex_conn.sds.get(filter_fields={'id': sds_id})
            if len(sds_details) == 0:
                msg = "SDS with identifier '%s' not found" % sds_id
                LOG.error(msg)
                return None
            if 'protectionDomainId' in sds_details[0] and sds_details[0]['protectionDomainId']:
                pd_details = self.get_protection_domain(protection_domain_id=sds_details[0]['protectionDomainId'])
                sds_details[0]['protectionDomainName'] = pd_details['name']
            if 'rmcacheSizeInKb' in sds_details[0] and sds_details[0]['rmcacheSizeInKb']:
                rmcache_size_mb = sds_details[0]['rmcacheSizeInKb'] / 1024
                sds_details[0]['rmcacheSizeInMb'] = int(rmcache_size_mb)
            if 'faultSetId' in sds_details[0] and sds_details[0]['faultSetId']:
                fs_details = self.get_fault_set(fault_set_id=sds_details[0]['faultSetId'], protection_domain_id=sds_details[0]['protectionDomainId'])
                sds_details[0]['faultSetName'] = fs_details['name']
            return sds_details[0]
        except Exception as e:
            error_msg = "Failed to get the SDS '%s' with error '%s'" % (sds_id, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def validate_parameters(self, sds_params):
        params = [sds_params['sds_name'], sds_params['sds_new_name']]
        for param in params:
            if param is not None and len(param.strip()) == 0:
                error_msg = 'Provide valid value for name for the creation/modification of the SDS.'
                LOG.error(error_msg)
                self.module.fail_json(msg=error_msg)