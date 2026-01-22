from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
class ElementOSVolumeClone(object):
    """
    Contains methods to parse arguments,
    derive details of Element Software objects
    and send requests to Element OS via
    the Solidfire SDK
    """

    def __init__(self):
        """
        Parse arguments, setup state variables,
        check paramenters and ensure SDK is installed
        """
        self._size_unit_map = netapp_utils.SF_BYTE_MAP
        self.argument_spec = netapp_utils.ontap_sf_host_argument_spec()
        self.argument_spec.update(dict(name=dict(required=True), src_volume_id=dict(required=True), src_snapshot_id=dict(), account_id=dict(required=True), attributes=dict(type='dict', default=None), size=dict(type='int'), size_unit=dict(default='gb', choices=['bytes', 'b', 'kb', 'mb', 'gb', 'tb', 'pb', 'eb', 'zb', 'yb'], type='str'), access=dict(type='str', default=None, choices=['readOnly', 'readWrite', 'locked', 'replicationTarget'])))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        parameters = self.module.params
        self.name = parameters['name']
        self.src_volume_id = parameters['src_volume_id']
        self.src_snapshot_id = parameters['src_snapshot_id']
        self.account_id = parameters['account_id']
        self.attributes = parameters['attributes']
        self.size_unit = parameters['size_unit']
        if parameters['size'] is not None:
            self.size = parameters['size'] * self._size_unit_map[self.size_unit]
        else:
            self.size = None
        self.access = parameters['access']
        if HAS_SF_SDK is False:
            self.module.fail_json(msg='Unable to import the SolidFire Python SDK')
        else:
            self.sfe = netapp_utils.create_sf_connection(module=self.module)
        self.elementsw_helper = NaElementSWModule(self.sfe)
        if self.attributes is not None:
            self.attributes.update(self.elementsw_helper.set_element_attributes(source='na_elementsw_volume_clone'))
        else:
            self.attributes = self.elementsw_helper.set_element_attributes(source='na_elementsw_volume_clone')

    def get_account_id(self):
        """
            Return account id if found
        """
        try:
            self.account_id = self.elementsw_helper.account_exists(self.account_id)
            return self.account_id
        except Exception as err:
            self.module.fail_json(msg='Error: account_id %s does not exist' % self.account_id, exception=to_native(err))

    def get_snapshot_id(self):
        """
            Return snapshot details if found
        """
        src_snapshot = self.elementsw_helper.get_snapshot(self.src_snapshot_id, self.src_volume_id)
        if src_snapshot is not None:
            self.src_snapshot_id = src_snapshot.snapshot_id
            return self.src_snapshot_id
        return None

    def get_src_volume_id(self):
        """
            Return volume id if found
        """
        src_vol_id = self.elementsw_helper.volume_exists(self.src_volume_id, self.account_id)
        if src_vol_id is not None:
            self.src_volume_id = src_vol_id
            return self.src_volume_id
        return None

    def clone_volume(self):
        """Clone Volume from source"""
        try:
            self.sfe.clone_volume(volume_id=self.src_volume_id, name=self.name, new_account_id=self.account_id, new_size=self.size, access=self.access, snapshot_id=self.src_snapshot_id, attributes=self.attributes)
        except Exception as err:
            self.module.fail_json(msg='Error creating clone %s of size %s' % (self.name, self.size), exception=to_native(err))

    def apply(self):
        """Perform pre-checks, call functions and exit"""
        changed = False
        result_message = ''
        if self.get_account_id() is None:
            self.module.fail_json(msg='Account id not found: %s' % self.account_id)
        if self.elementsw_helper.volume_exists(self.name, self.account_id) is None:
            if self.get_src_volume_id() is not None:
                if self.src_snapshot_id and (not self.get_snapshot_id()):
                    self.module.fail_json(msg='Snapshot id not found: %s' % self.src_snapshot_id)
                changed = True
            else:
                self.module.fail_json(msg='Volume id not found %s' % self.src_volume_id)
        if changed:
            if self.module.check_mode:
                result_message = 'Check mode, skipping changes'
            else:
                self.clone_volume()
                result_message = 'Volume cloned'
        self.module.exit_json(changed=changed, msg=result_message)