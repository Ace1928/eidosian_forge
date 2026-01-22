from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
class PowerFlexStoragePool(object):
    """Class with StoragePool operations"""

    def __init__(self):
        """ Define all parameters required by this module"""
        self.module_params = utils.get_powerflex_gateway_host_parameters()
        self.module_params.update(get_powerflex_storagepool_parameters())
        ' initialize the ansible module '
        mut_ex_args = [['storage_pool_name', 'storage_pool_id'], ['protection_domain_name', 'protection_domain_id'], ['storage_pool_id', 'protection_domain_name'], ['storage_pool_id', 'protection_domain_id']]
        required_one_of_args = [['storage_pool_name', 'storage_pool_id']]
        self.module = AnsibleModule(argument_spec=self.module_params, supports_check_mode=False, mutually_exclusive=mut_ex_args, required_one_of=required_one_of_args)
        utils.ensure_required_libs(self.module)
        try:
            self.powerflex_conn = utils.get_powerflex_gateway_host_connection(self.module.params)
            LOG.info('Got the PowerFlex system connection object instance')
        except Exception as e:
            LOG.error(str(e))
            self.module.fail_json(msg=str(e))

    def get_protection_domain(self, protection_domain_name=None, protection_domain_id=None):
        """Get protection domain details
            :param protection_domain_name: Name of the protection domain
            :param protection_domain_id: ID of the protection domain
            :return: Protection domain details
        """
        name_or_id = protection_domain_id if protection_domain_id else protection_domain_name
        try:
            filter_fields = {}
            if protection_domain_id:
                filter_fields = {'id': protection_domain_id}
            if protection_domain_name:
                filter_fields = {'name': protection_domain_name}
            pd_details = self.powerflex_conn.protection_domain.get(filter_fields=filter_fields)
            if pd_details:
                return pd_details[0]
            if not pd_details:
                err_msg = 'Unable to find the protection domain with {0}. Please enter a valid protection domain name/id.'.format(name_or_id)
                self.module.fail_json(msg=err_msg)
        except Exception as e:
            errormsg = 'Failed to get the protection domain {0} with error {1}'.format(name_or_id, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def get_storage_pool(self, storage_pool_id=None, storage_pool_name=None, pd_id=None):
        """Get storage pool details
            :param pd_id: ID of the protection domain
            :param storage_pool_name: The name of the storage pool
            :param storage_pool_id: The storage pool id
            :return: Storage pool details
        """
        name_or_id = storage_pool_id if storage_pool_id else storage_pool_name
        try:
            filter_fields = {}
            if storage_pool_id:
                filter_fields = {'id': storage_pool_id}
            if storage_pool_name:
                filter_fields.update({'name': storage_pool_name})
            if pd_id:
                filter_fields.update({'protectionDomainId': pd_id})
            pool_details = self.powerflex_conn.storage_pool.get(filter_fields=filter_fields)
            if pool_details:
                if len(pool_details) > 1:
                    err_msg = 'More than one storage pool found with {0}, Please provide protection domain Name/Id to fetch the unique storage pool'.format(storage_pool_name)
                    LOG.error(err_msg)
                    self.module.fail_json(msg=err_msg)
                elif len(pool_details) == 1:
                    pool_details = pool_details[0]
                    statistics = self.powerflex_conn.storage_pool.get_statistics(pool_details['id'])
                    pool_details['statistics'] = statistics if statistics else {}
                    pd_id = pool_details['protectionDomainId']
                    pd_name = self.get_protection_domain(protection_domain_id=pd_id)['name']
                    pool_details['protectionDomainName'] = pd_name
                else:
                    pool_details = None
            return pool_details
        except Exception as e:
            errormsg = 'Failed to get the storage pool {0} with error {1}'.format(name_or_id, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def create_storage_pool(self, pool_name, pd_id, media_type, use_rfcache=None, use_rmcache=None):
        """
        Create a storage pool
        :param pool_name: Name of the storage pool
        :param pd_id: ID of the storage pool
        :param media_type: Type of storage device in the pool
        :param use_rfcache: Enable/Disable RFcache on pool
        :param use_rmcache: Enable/Disable RMcache on pool
        :return: True, if the operation is successful
        """
        try:
            if media_type == 'Transitional':
                self.module.fail_json(msg='TRANSITIONAL media type is not supported during creation. Please enter a valid media type')
            if pd_id is None:
                self.module.fail_json(msg='Please provide protection domain details for creation of a storage pool')
            self.powerflex_conn.storage_pool.create(media_type=media_type, protection_domain_id=pd_id, name=pool_name, use_rfcache=use_rfcache, use_rmcache=use_rmcache)
            return True
        except Exception as e:
            errormsg = 'Failed to create the storage pool {0} with error {1}'.format(pool_name, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def modify_storage_pool(self, pool_id, modify_dict):
        """
        Modify the parameters of the storage pool.
        :param modify_dict: Dict containing parameters which are to be
         modified
        :param pool_id: Id of the pool.
        :return: True, if the operation is successful.
        """
        try:
            if 'new_name' in modify_dict:
                self.powerflex_conn.storage_pool.rename(pool_id, modify_dict['new_name'])
            if 'use_rmcache' in modify_dict:
                self.powerflex_conn.storage_pool.set_use_rmcache(pool_id, modify_dict['use_rmcache'])
            if 'use_rfcache' in modify_dict:
                self.powerflex_conn.storage_pool.set_use_rfcache(pool_id, modify_dict['use_rfcache'])
            if 'media_type' in modify_dict:
                self.powerflex_conn.storage_pool.set_media_type(pool_id, modify_dict['media_type'])
            return True
        except Exception as e:
            err_msg = 'Failed to update the storage pool {0} with error {1}'.format(pool_id, str(e))
            LOG.error(err_msg)
            self.module.fail_json(msg=err_msg)

    def verify_params(self, pool_details, pd_name, pd_id):
        """
        :param pool_details: Details of the storage pool
        :param pd_name: Name of the protection domain
        :param pd_id: Id of the protection domain
        """
        if pd_id and pd_id != pool_details['protectionDomainId']:
            self.module.fail_json(msg="Entered protection domain id does not match with the storage pool's protection domain id. Please enter a correct protection domain id.")
        if pd_name and pd_name != pool_details['protectionDomainName']:
            self.module.fail_json(msg="Entered protection domain name does not match with the storage pool's protection domain name. Please enter a correct protection domain name.")

    def perform_module_operation(self):
        """ Perform different actions on Storage Pool based on user input
            in the playbook """
        pool_name = self.module.params['storage_pool_name']
        pool_id = self.module.params['storage_pool_id']
        pool_new_name = self.module.params['storage_pool_new_name']
        state = self.module.params['state']
        pd_name = self.module.params['protection_domain_name']
        pd_id = self.module.params['protection_domain_id']
        use_rmcache = self.module.params['use_rmcache']
        use_rfcache = self.module.params['use_rfcache']
        media_type = self.module.params['media_type']
        if media_type == 'TRANSITIONAL':
            media_type = 'Transitional'
        result = dict(storage_pool_details={})
        changed = False
        pd_details = None
        if pd_name or pd_id:
            pd_details = self.get_protection_domain(protection_domain_id=pd_id, protection_domain_name=pd_name)
        if pd_details:
            pd_id = pd_details['id']
        if pool_name is not None and len(pool_name.strip()) == 0:
            self.module.fail_json(msg='Empty or white spaced string provided in storage_pool_name. Please provide valid storage pool name.')
        pool_details = self.get_storage_pool(storage_pool_id=pool_id, storage_pool_name=pool_name, pd_id=pd_id)
        if pool_name and pool_details:
            pool_id = pool_details['id']
            self.verify_params(pool_details, pd_name, pd_id)
        if state == 'present' and (not pool_details):
            LOG.info('Creating new storage pool')
            if pool_id:
                self.module.fail_json(msg='storage_pool_name is missing & name required to create a storage pool. Please enter a valid storage_pool_name.')
            if pool_new_name is not None:
                self.module.fail_json(msg='storage_pool_new_name is passed during creation. storage_pool_new_name is not allowed during creation of a storage pool.')
            changed = self.create_storage_pool(pool_name, pd_id, media_type, use_rfcache, use_rmcache)
            if changed:
                pool_id = self.get_storage_pool(storage_pool_id=pool_id, storage_pool_name=pool_name, pd_id=pd_id)['id']
        if state == 'present' and pool_details:
            if pool_new_name is not None and len(pool_new_name.strip()) == 0:
                self.module.fail_json(msg='Empty/White spaced name is not allowed during renaming of a storage pool. Please enter a valid storage pool new name.')
            modify_dict = to_modify(pool_details, use_rmcache, use_rfcache, pool_new_name, media_type)
            if bool(modify_dict):
                LOG.info('Modify attributes of storage pool')
                changed = self.modify_storage_pool(pool_id, modify_dict)
        if state == 'absent' and pool_details:
            msg = 'Deleting storage pool is not supported through ansible module.'
            LOG.error(msg)
            self.module.fail_json(msg=msg)
        if state == 'present':
            pool_details = self.get_storage_pool(storage_pool_id=pool_id)
            pd_id = pool_details['protectionDomainId']
            pd_name = self.get_protection_domain(protection_domain_id=pd_id)['name']
            pool_details['protectionDomainName'] = pd_name
            result['storage_pool_details'] = pool_details
        result['changed'] = changed
        self.module.exit_json(**result)