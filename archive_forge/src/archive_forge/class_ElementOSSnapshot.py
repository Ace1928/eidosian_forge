from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
class ElementOSSnapshot(object):
    """
    Element OS Snapshot Manager
    """

    def __init__(self):
        self.argument_spec = netapp_utils.ontap_sf_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, choices=['present', 'absent'], default='present'), account_id=dict(required=True, type='str'), name=dict(required=False, type='str'), src_volume_id=dict(required=True, type='str'), retention=dict(required=False, type='str'), src_snapshot_id=dict(required=False, type='str'), enable_remote_replication=dict(required=False, type='bool'), expiration_time=dict(required=False, type='str'), snap_mirror_label=dict(required=False, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        input_params = self.module.params
        self.state = input_params['state']
        self.name = input_params['name']
        self.account_id = input_params['account_id']
        self.src_volume_id = input_params['src_volume_id']
        self.src_snapshot_id = input_params['src_snapshot_id']
        self.retention = input_params['retention']
        self.properties_provided = False
        self.expiration_time = input_params['expiration_time']
        if input_params['expiration_time'] is not None:
            self.properties_provided = True
        self.enable_remote_replication = input_params['enable_remote_replication']
        if input_params['enable_remote_replication'] is not None:
            self.properties_provided = True
        self.snap_mirror_label = input_params['snap_mirror_label']
        if input_params['snap_mirror_label'] is not None:
            self.properties_provided = True
        if self.state == 'absent' and self.src_snapshot_id is None:
            self.module.fail_json(msg='Please provide required parameter : snapshot_id')
        if HAS_SF_SDK is False:
            self.module.fail_json(msg='Unable to import the SolidFire Python SDK')
        else:
            self.sfe = netapp_utils.create_sf_connection(module=self.module)
        self.elementsw_helper = NaElementSWModule(self.sfe)
        self.attributes = self.elementsw_helper.set_element_attributes(source='na_elementsw_snapshot')

    def get_account_id(self):
        """
            Return account id if found
        """
        try:
            self.account_id = self.elementsw_helper.account_exists(self.account_id)
            return self.account_id
        except Exception as err:
            self.module.fail_json(msg='Error: account_id %s does not exist' % self.account_id, exception=to_native(err))

    def get_src_volume_id(self):
        """
            Return volume id if found
        """
        src_vol_id = self.elementsw_helper.volume_exists(self.src_volume_id, self.account_id)
        if src_vol_id is not None:
            self.src_volume_id = src_vol_id
            return self.src_volume_id
        return None

    def get_snapshot(self, name=None):
        """
            Return snapshot details if found
        """
        src_snapshot = None
        if name is not None:
            src_snapshot = self.elementsw_helper.get_snapshot(name, self.src_volume_id)
        elif self.src_snapshot_id is not None:
            src_snapshot = self.elementsw_helper.get_snapshot(self.src_snapshot_id, self.src_volume_id)
        if src_snapshot is not None:
            self.src_snapshot_id = src_snapshot.snapshot_id
        return src_snapshot

    def create_snapshot(self):
        """
        Create Snapshot
        """
        try:
            self.sfe.create_snapshot(volume_id=self.src_volume_id, snapshot_id=self.src_snapshot_id, name=self.name, enable_remote_replication=self.enable_remote_replication, retention=self.retention, snap_mirror_label=self.snap_mirror_label, attributes=self.attributes)
        except Exception as exception_object:
            self.module.fail_json(msg='Error creating snapshot %s' % to_native(exception_object), exception=traceback.format_exc())

    def modify_snapshot(self):
        """
        Modify Snapshot Properties
        """
        try:
            self.sfe.modify_snapshot(snapshot_id=self.src_snapshot_id, expiration_time=self.expiration_time, enable_remote_replication=self.enable_remote_replication, snap_mirror_label=self.snap_mirror_label)
        except Exception as exception_object:
            self.module.fail_json(msg='Error modify snapshot %s' % to_native(exception_object), exception=traceback.format_exc())

    def delete_snapshot(self):
        """
        Delete Snapshot
        """
        try:
            self.sfe.delete_snapshot(snapshot_id=self.src_snapshot_id)
        except Exception as exception_object:
            self.module.fail_json(msg='Error delete snapshot %s' % to_native(exception_object), exception=traceback.format_exc())

    def apply(self):
        """
        Check, process and initiate snapshot operation
        """
        changed = False
        result_message = None
        self.get_account_id()
        if self.get_src_volume_id() is None:
            self.module.fail_json(msg='Volume id not found %s' % self.src_volume_id)
        snapshot_detail = self.get_snapshot()
        if snapshot_detail:
            if self.properties_provided:
                if self.expiration_time != snapshot_detail.expiration_time:
                    changed = True
                else:
                    self.expiration_time = snapshot_detail.expiration_time
                if self.enable_remote_replication != snapshot_detail.enable_remote_replication:
                    changed = True
                else:
                    self.enable_remote_replication = snapshot_detail.enable_remote_replication
                if self.snap_mirror_label != snapshot_detail.snap_mirror_label:
                    changed = True
                else:
                    self.snap_mirror_label = snapshot_detail.snap_mirror_label
        if self.account_id is None or self.src_volume_id is None or self.module.check_mode:
            changed = False
            result_message = 'Check mode, skipping changes'
        elif self.state == 'absent' and snapshot_detail is not None:
            self.delete_snapshot()
            changed = True
        elif self.state == 'present' and snapshot_detail is not None:
            if changed:
                self.modify_snapshot()
            elif not self.properties_provided:
                if self.name is not None:
                    snapshot = self.get_snapshot(self.name)
                    if snapshot is None:
                        self.create_snapshot()
                        changed = True
                else:
                    self.create_snapshot()
                    changed = True
        elif self.state == 'present':
            if self.name is not None:
                snapshot = self.get_snapshot(self.name)
                if snapshot is None:
                    self.create_snapshot()
                    changed = True
            else:
                self.create_snapshot()
                changed = True
        else:
            changed = False
            result_message = 'No changes requested, skipping changes'
        self.module.exit_json(changed=changed, msg=result_message)