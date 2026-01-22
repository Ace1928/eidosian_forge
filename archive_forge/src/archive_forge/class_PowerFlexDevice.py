from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
class PowerFlexDevice(object):
    """Class with device operations"""

    def __init__(self):
        """ Define all parameters required by this module"""
        self.module_params = utils.get_powerflex_gateway_host_parameters()
        self.module_params.update(get_powerflex_device_parameters())
        mut_ex_args = [['sds_name', 'sds_id'], ['device_name', 'device_id'], ['protection_domain_name', 'protection_domain_id'], ['storage_pool_name', 'storage_pool_id'], ['acceleration_pool_name', 'acceleration_pool_id'], ['acceleration_pool_id', 'storage_pool_id'], ['acceleration_pool_name', 'storage_pool_name'], ['device_id', 'sds_name'], ['device_id', 'sds_id'], ['device_id', 'current_pathname']]
        self.module = AnsibleModule(argument_spec=self.module_params, supports_check_mode=False, mutually_exclusive=mut_ex_args)
        utils.ensure_required_libs(self.module)
        try:
            self.powerflex_conn = utils.get_powerflex_gateway_host_connection(self.module.params)
            LOG.info('Got the PowerFlex system connection object instance')
        except Exception as e:
            LOG.error(str(e))
            self.module.fail_json(msg=str(e))

    def get_device_details(self, current_pathname=None, sds_id=None, device_name=None, device_id=None):
        """Get device details
            :param current_pathname: Device path name
            :type current_pathname: str
            :param sds_id: ID of the SDS
            :type sds_id: str
            :param device_name: Name of the device
            :type device_name: str
            :param device_id: ID of the device
            :type device_id: str
            :return: Details of device if it exist
            :rtype: dict
        """
        try:
            if current_pathname and sds_id:
                device_details = self.powerflex_conn.device.get(filter_fields={'deviceCurrentPathName': current_pathname, 'sdsId': sds_id})
            elif device_name and sds_id:
                device_details = self.powerflex_conn.device.get(filter_fields={'name': device_name, 'sdsId': sds_id})
            else:
                device_details = self.powerflex_conn.device.get(filter_fields={'id': device_id})
            if len(device_details) == 0:
                msg = 'Device not found'
                LOG.info(msg)
                return None
            return device_details[0]
        except Exception as e:
            error_msg = "Failed to get the device with error '%s'" % str(e)
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def get_sds(self, sds_name=None, sds_id=None):
        """Get SDS details
            :param sds_name: Name of the SDS
            :param sds_id: ID of the SDS
            :return: SDS details
            :rtype: dict
        """
        name_or_id = sds_id if sds_id else sds_name
        try:
            sds_details = None
            if sds_id:
                sds_details = self.powerflex_conn.sds.get(filter_fields={'id': sds_id})
            if sds_name:
                sds_details = self.powerflex_conn.sds.get(filter_fields={'name': sds_name})
            if not sds_details:
                error_msg = "Unable to find the SDS with '%s'. Please enter a valid SDS name/id." % name_or_id
                LOG.error(error_msg)
                self.module.fail_json(msg=error_msg)
            return sds_details[0]
        except Exception as e:
            error_msg = "Failed to get the SDS '%s' with error '%s'" % (name_or_id, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def get_protection_domain(self, protection_domain_name=None, protection_domain_id=None):
        """Get protection domain details
            :param protection_domain_name: Name of the protection domain
            :param protection_domain_id: ID of the protection domain
            :return: Protection domain details
            :rtype: dict
        """
        name_or_id = protection_domain_id if protection_domain_id else protection_domain_name
        try:
            pd_details = None
            if protection_domain_id:
                pd_details = self.powerflex_conn.protection_domain.get(filter_fields={'id': protection_domain_id})
            if protection_domain_name:
                pd_details = self.powerflex_conn.protection_domain.get(filter_fields={'name': protection_domain_name})
            if not pd_details:
                error_msg = "Unable to find the protection domain with '%s'. Please enter a valid protection domain name/id." % name_or_id
                LOG.error(error_msg)
                self.module.fail_json(msg=error_msg)
            return pd_details[0]
        except Exception as e:
            error_msg = "Failed to get the protection domain '%s' with error '%s'" % (name_or_id, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def get_storage_pool(self, storage_pool_name=None, storage_pool_id=None, protection_domain_id=None):
        """Get storage pool details
            :param storage_pool_name: Name of the storage pool
            :param storage_pool_id: ID of the storage pool
            :param protection_domain_id: ID of the protection domain
            :return: Storage pool details
            :rtype: dict
        """
        name_or_id = storage_pool_id if storage_pool_id else storage_pool_name
        try:
            storage_pool_details = None
            if storage_pool_id:
                storage_pool_details = self.powerflex_conn.storage_pool.get(filter_fields={'id': storage_pool_id})
            if storage_pool_name:
                storage_pool_details = self.powerflex_conn.storage_pool.get(filter_fields={'name': storage_pool_name, 'protectionDomainId': protection_domain_id})
            if not storage_pool_details:
                error_msg = "Unable to find the storage pool with '%s'. Please enter a valid storage pool name/id." % name_or_id
                LOG.error(error_msg)
                self.module.fail_json(msg=error_msg)
            return storage_pool_details[0]
        except Exception as e:
            error_msg = "Failed to get the storage_pool '%s' with error '%s'" % (name_or_id, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def get_acceleration_pool(self, acceleration_pool_name=None, acceleration_pool_id=None, protection_domain_id=None):
        """Get acceleration pool details
            :param acceleration_pool_name: Name of the acceleration pool
            :param acceleration_pool_id: ID of the acceleration pool
            :param protection_domain_id: ID of the protection domain
            :return: Acceleration pool details
            :rtype: dict
        """
        name_or_id = acceleration_pool_id if acceleration_pool_id else acceleration_pool_name
        try:
            acceleration_pool_details = None
            if acceleration_pool_id:
                acceleration_pool_details = self.powerflex_conn.acceleration_pool.get(filter_fields={'id': acceleration_pool_id})
            if acceleration_pool_name:
                acceleration_pool_details = self.powerflex_conn.acceleration_pool.get(filter_fields={'name': acceleration_pool_name, 'protectionDomainId': protection_domain_id})
            if not acceleration_pool_details:
                error_msg = "Unable to find the acceleration pool with '%s'. Please enter a valid acceleration pool name/id." % name_or_id
                LOG.error(error_msg)
                self.module.fail_json(msg=error_msg)
            return acceleration_pool_details[0]
        except Exception as e:
            error_msg = "Failed to get the acceleration pool '%s' with error '%s'" % (name_or_id, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def add_device(self, device_name, current_pathname, sds_id, storage_pool_id, media_type, acceleration_pool_id, external_acceleration_type):
        """Add device
            :param device_name: Device name
            :type device_name: str
            :param current_pathname: Current pathname of device
            :type current_pathname: str
            :param sds_id: SDS ID
            :type sds_id: str
            :param storage_pool_id: Storage Pool ID
            :type storage_pool_id: str
            :param media_type: Media type of device
            :type media_type: str
            :param acceleration_pool_id: Acceleration pool ID
            :type acceleration_pool_id: str
            :param external_acceleration_type: External acceleration type
            :type external_acceleration_type: str
            return: Boolean indicating if add device operation is successful
        """
        try:
            if device_name is None or len(device_name.strip()) == 0:
                error_msg = 'Please provide valid device_name value for adding a device.'
                LOG.error(error_msg)
                self.module.fail_json(msg=error_msg)
            if current_pathname is None or len(current_pathname.strip()) == 0:
                error_msg = 'Current pathname of device is a mandatory parameter for adding a device. Please enter a valid value.'
                LOG.error(error_msg)
                self.module.fail_json(msg=error_msg)
            if sds_id is None or len(sds_id.strip()) == 0:
                error_msg = 'Please provide valid sds_id value for adding a device.'
                LOG.error(error_msg)
                self.module.fail_json(msg=error_msg)
            if storage_pool_id is None and acceleration_pool_id is None:
                error_msg = 'Please provide either storage pool name/ID or acceleration pool name/ID for adding a device.'
                LOG.error(error_msg)
                self.module.fail_json(msg=error_msg)
            add_params = 'current_pathname: %s, sds_id: %s, acceleration_pool_id: %s,external_acceleration_type: %s,media_type: %s,device_name: %s,storage_pool_id: %s,' % (current_pathname, sds_id, acceleration_pool_id, external_acceleration_type, media_type, device_name, storage_pool_id)
            LOG.info('Adding device with params: %s', add_params)
            self.powerflex_conn.device.create(current_pathname=current_pathname, sds_id=sds_id, acceleration_pool_id=acceleration_pool_id, external_acceleration_type=external_acceleration_type, media_type=media_type, name=device_name, storage_pool_id=storage_pool_id, force=self.module.params['force'])
            return True
        except Exception as e:
            error_msg = "Adding device %s operation failed with error '%s'" % (device_name, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def remove_device(self, device_id):
        """Remove device
            :param device_id: Device ID
            :type device_id: str
            return: Boolean indicating if remove device operation is
                    successful
        """
        try:
            LOG.info('Device to be removed: %s', device_id)
            self.powerflex_conn.device.delete(device_id=device_id)
            return True
        except Exception as e:
            error_msg = "Remove device '%s' operation failed with error '%s'" % (device_id, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def validate_input_parameters(self, device_name=None, device_id=None, current_pathname=None, sds_name=None, sds_id=None):
        """Validate the input parameters"""
        if current_pathname:
            if (sds_name is None or len(sds_name.strip()) == 0) and (sds_id is None or len(sds_id.strip()) == 0):
                error_msg = 'sds_name or sds_id is mandatory along with current_pathname. Please enter a valid value.'
                LOG.error(error_msg)
                self.module.fail_json(msg=error_msg)
        elif current_pathname is not None and len(current_pathname.strip()) == 0:
            error_msg = 'Please enter a valid value for current_pathname.'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)
        if device_name:
            if (sds_name is None or len(sds_name.strip()) == 0) and (sds_id is None or len(sds_id.strip()) == 0):
                error_msg = 'sds_name or sds_id is mandatory along with device_name. Please enter a valid value.'
                LOG.error(error_msg)
                self.module.fail_json(msg=error_msg)
        elif device_name is not None and len(device_name.strip()) == 0:
            error_msg = 'Please enter a valid value for device_name.'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)
        if sds_name:
            if (current_pathname is None or len(current_pathname.strip()) == 0) and (device_name is None or len(device_name.strip()) == 0):
                error_msg = 'current_pathname or device_name is mandatory along with sds_name. Please enter a valid value.'
                LOG.error(error_msg)
                self.module.fail_json(msg=error_msg)
        elif sds_name is not None and len(sds_name.strip()) == 0:
            error_msg = 'Please enter a valid value for sds_name.'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)
        if sds_id:
            if (current_pathname is None or len(current_pathname.strip()) == 0) and (device_name is None or len(device_name.strip()) == 0):
                error_msg = 'current_pathname or device_name is mandatory along with sds_id. Please enter a valid value.'
                LOG.error(error_msg)
                self.module.fail_json(msg=error_msg)
        elif sds_id is not None and len(sds_id.strip()) == 0:
            error_msg = 'Please enter a valid value for sds_id.'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)
        if device_id is not None and len(device_id.strip()) == 0:
            error_msg = 'Please provide valid device_id value to identify a device.'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)
        if current_pathname is None and device_name is None and (device_id is None):
            error_msg = 'Please specify a valid parameter combination to identify a device.'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def validate_add_parameters(self, device_id=None, external_acceleration_type=None, storage_pool_id=None, storage_pool_name=None, acceleration_pool_id=None, acceleration_pool_name=None):
        """Validate the add device parameters"""
        if device_id:
            error_msg = 'Addition of device is allowed using device_name only, device_id given.'
            LOG.info(error_msg)
            self.module.fail_json(msg=error_msg)
        if external_acceleration_type and storage_pool_id is None and (storage_pool_name is None) and (acceleration_pool_id is None) and (acceleration_pool_name is None):
            error_msg = 'Storage Pool ID/name or Acceleration Pool ID/name is mandatory along with external_acceleration_type.'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def perform_module_operation(self):
        """
        Perform different actions on device based on parameters passed in
        the playbook
        """
        current_pathname = self.module.params['current_pathname']
        device_name = self.module.params['device_name']
        device_id = self.module.params['device_id']
        sds_name = self.module.params['sds_name']
        sds_id = self.module.params['sds_id']
        storage_pool_name = self.module.params['storage_pool_name']
        storage_pool_id = self.module.params['storage_pool_id']
        acceleration_pool_id = self.module.params['acceleration_pool_id']
        acceleration_pool_name = self.module.params['acceleration_pool_name']
        protection_domain_name = self.module.params['protection_domain_name']
        protection_domain_id = self.module.params['protection_domain_id']
        external_acceleration_type = self.module.params['external_acceleration_type']
        media_type = self.module.params['media_type']
        state = self.module.params['state']
        changed = False
        result = dict(changed=False, device_details={})
        self.validate_input_parameters(device_name, device_id, current_pathname, sds_name, sds_id)
        if sds_name:
            sds_details = self.get_sds(sds_name)
            if sds_details:
                sds_id = sds_details['id']
            msg = "Fetched the SDS details with id '%s', name '%s'" % (sds_id, sds_name)
            LOG.info(msg)
        device_details = self.get_device_details(current_pathname, sds_id, device_name, device_id)
        if device_details:
            device_id = device_details['id']
        msg = 'Fetched the device details %s' % str(device_details)
        LOG.info(msg)
        add_changed = False
        if state == 'present' and (not device_details):
            if protection_domain_name and (storage_pool_name or acceleration_pool_name):
                pd_details = self.get_protection_domain(protection_domain_name)
                if pd_details:
                    protection_domain_id = pd_details['id']
                msg = "Fetched the protection domain details with id '%s', name '%s'" % (protection_domain_id, protection_domain_name)
                LOG.info(msg)
            if storage_pool_name:
                if protection_domain_id:
                    storage_pool_details = self.get_storage_pool(storage_pool_name=storage_pool_name, protection_domain_id=protection_domain_id)
                    if storage_pool_details:
                        storage_pool_id = storage_pool_details['id']
                    msg = "Fetched the storage pool details with id '%s', name '%s'" % (storage_pool_id, storage_pool_name)
                    LOG.info(msg)
                else:
                    error_msg = 'Protection domain name/id is required to uniquely identify a storage pool, only storage_pool_name is given.'
                    LOG.info(error_msg)
                    self.module.fail_json(msg=error_msg)
            if acceleration_pool_name:
                if protection_domain_id:
                    acceleration_pool_details = self.get_acceleration_pool(acceleration_pool_name=acceleration_pool_name, protection_domain_id=protection_domain_id)
                    if acceleration_pool_details:
                        acceleration_pool_id = acceleration_pool_details['id']
                    msg = "Fetched the acceleration pool details with id '%s', name '%s'" % (acceleration_pool_id, acceleration_pool_name)
                    LOG.info(msg)
                else:
                    error_msg = 'Protection domain name/id is required to uniquely identify a acceleration pool, only acceleration_pool_name is given.'
                    LOG.info(error_msg)
                    self.module.fail_json(msg=error_msg)
            self.validate_add_parameters(device_id, external_acceleration_type, storage_pool_id, storage_pool_name, acceleration_pool_id, acceleration_pool_name)
            add_changed = self.add_device(device_name, current_pathname, sds_id, storage_pool_id, media_type, acceleration_pool_id, external_acceleration_type)
            if add_changed:
                device_details = self.get_device_details(device_name=device_name, sds_id=sds_id)
                device_id = device_details['id']
                msg = 'Device created successfully, fetched device details %s' % str(device_details)
                LOG.info(msg)
        remove_changed = False
        if state == 'absent' and device_details:
            remove_changed = self.remove_device(device_id)
        if add_changed or remove_changed:
            changed = True
        if device_details and state == 'present':
            modify_dict = to_modify(device_details, media_type, external_acceleration_type)
            if modify_dict:
                error_msg = 'Modification of device attributes is currently not supported by Ansible modules.'
                LOG.info(error_msg)
                self.module.fail_json(msg=error_msg)
        if state == 'present':
            device_details = self.show_output(device_id)
            result['device_details'] = device_details
        result['changed'] = changed
        self.module.exit_json(**result)

    def show_output(self, device_id):
        """Show device details
            :param device_id: ID of the device
            :type device_id: str
            :return: Details of device
            :rtype: dict
        """
        try:
            device_details = self.powerflex_conn.device.get(filter_fields={'id': device_id})
            if len(device_details) == 0:
                msg = "Device with identifier '%s' not found" % device_id
                LOG.error(msg)
                return None
            if 'sdsId' in device_details[0] and device_details[0]['sdsId']:
                sds_details = self.get_sds(sds_id=device_details[0]['sdsId'])
                device_details[0]['sdsName'] = sds_details['name']
            if 'storagePoolId' in device_details[0] and device_details[0]['storagePoolId']:
                sp_details = self.get_storage_pool(storage_pool_id=device_details[0]['storagePoolId'])
                device_details[0]['storagePoolName'] = sp_details['name']
                pd_id = sp_details['protectionDomainId']
                device_details[0]['protectionDomainId'] = pd_id
                pd_details = self.get_protection_domain(protection_domain_id=pd_id)
                device_details[0]['protectionDomainName'] = pd_details['name']
            if 'accelerationPoolId' in device_details[0] and device_details[0]['accelerationPoolId']:
                ap_details = self.get_acceleration_pool(acceleration_pool_id=device_details[0]['accelerationPoolId'])
                device_details[0]['accelerationPoolName'] = ap_details['name']
                pd_id = ap_details['protectionDomainId']
                device_details[0]['protectionDomainId'] = pd_id
                pd_details = self.get_protection_domain(protection_domain_id=pd_id)
                device_details[0]['protectionDomainName'] = pd_details['name']
            return device_details[0]
        except Exception as e:
            error_msg = "Failed to get the device '%s' with error '%s'" % (device_id, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)