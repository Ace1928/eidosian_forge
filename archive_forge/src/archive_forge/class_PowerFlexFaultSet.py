from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell import (
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
from ansible.module_utils.basic import AnsibleModule
class PowerFlexFaultSet(PowerFlexBase):
    """Class with FaultSet operations"""

    def __init__(self):
        """Define all parameters required by this module"""
        mutually_exclusive = [['fault_set_name', 'fault_set_id'], ['protection_domain_name', 'protection_domain_id']]
        required_one_of = [['fault_set_name', 'fault_set_id']]
        ansible_module_params = {'argument_spec': get_powerflex_fault_set_parameters(), 'supports_check_mode': True, 'mutually_exclusive': mutually_exclusive, 'required_one_of': required_one_of}
        super().__init__(AnsibleModule, ansible_module_params)
        self.result = dict(changed=False, fault_set_details={})

    def get_protection_domain(self, protection_domain_name=None, protection_domain_id=None):
        """Get the details of a protection domain in a given PowerFlex storage
        system"""
        return Configuration(self.powerflex_conn, self.module).get_protection_domain(protection_domain_name=protection_domain_name, protection_domain_id=protection_domain_id)

    def get_associated_sds(self, fault_set_id=None):
        """Get the details of SDS associated to given fault set in a given PowerFlex storage
        system"""
        return Configuration(self.powerflex_conn, self.module).get_associated_sds(fault_set_id=fault_set_id)

    def create_fault_set(self, fault_set_name, protection_domain_id):
        """
        Create Fault Set
        :param fault_set_name: Name of the fault set
        :type fault_set_name: str
        :param protection_domain_id: ID of the protection domain
        :type protection_domain_id: str
        :return: Boolean indicating if create operation is successful
        """
        try:
            if not self.module.check_mode:
                msg = f'Creating fault set with name: {fault_set_name} on protection domain with id: {protection_domain_id}'
                LOG.info(msg)
                self.powerflex_conn.fault_set.create(name=fault_set_name, protection_domain_id=protection_domain_id)
            return self.get_fault_set(fault_set_name=fault_set_name, protection_domain_id=protection_domain_id)
        except Exception as e:
            error_msg = f'Create fault set {fault_set_name} operation failed with error {str(e)}'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def get_fault_set(self, fault_set_name=None, fault_set_id=None, protection_domain_id=None):
        """Get fault set details
            :param fault_set_name: Name of the fault set
            :param fault_set_id: Id of the fault set
            :param protection_domain_id: ID of the protection domain
            :return: Fault set details
            :rtype: dict
        """
        return Configuration(self.powerflex_conn, self.module).get_fault_set(fault_set_name=fault_set_name, fault_set_id=fault_set_id, protection_domain_id=protection_domain_id)

    def is_rename_required(self, fault_set_details, fault_set_params):
        """To get the details of the fields to be modified."""
        if fault_set_params['fault_set_new_name'] is not None and fault_set_params['fault_set_new_name'] != fault_set_details['name']:
            return True
        return False

    def rename_fault_set(self, fault_set_id, new_name):
        """Perform rename operation on a fault set"""
        try:
            if not self.module.check_mode:
                self.powerflex_conn.fault_set.rename(fault_set_id=fault_set_id, name=new_name)
            return self.get_fault_set(fault_set_id=fault_set_id)
        except Exception as e:
            msg = f'Failed to rename the fault set instance with error {str(e)}'
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def delete_fault_set(self, fault_set_id):
        """Delete the Fault Set"""
        try:
            if not self.module.check_mode:
                LOG.info(msg=f'Removing Fault Set {fault_set_id}')
                self.powerflex_conn.fault_set.delete(fault_set_id)
                LOG.info('returning None')
                return None
            return self.get_fault_set(fault_set_id=fault_set_id)
        except Exception as e:
            errormsg = f'Removing Fault Set {fault_set_id} failed with error {str(e)}'
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def validate_parameters(self, fault_set_params):
        params = [fault_set_params['fault_set_name'], fault_set_params['fault_set_new_name']]
        for param in params:
            if param is not None and len(param.strip()) == 0:
                error_msg = 'Provide valid value for name for the creation/modification of the fault set.'
                LOG.error(error_msg)
                self.module.fail_json(msg=error_msg)
        if fault_set_params['fault_set_name'] is not None and fault_set_params['protection_domain_id'] is None and (fault_set_params['protection_domain_name'] is None):
            error_msg = 'Provide protection_domain_id/protection_domain_name with fault_set_name.'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)