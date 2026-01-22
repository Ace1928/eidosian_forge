from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
class ElementSWClusterConfig(object):
    """
    Element Software Configure Element SW Cluster
    """

    def __init__(self):
        self.argument_spec = netapp_utils.ontap_sf_host_argument_spec()
        self.argument_spec.update(dict(modify_cluster_full_threshold=dict(type='dict', options=dict(stage2_aware_threshold=dict(type='int', default=None), stage3_block_threshold_percent=dict(type='int', default=None), max_metadata_over_provision_factor=dict(type='int', default=None))), encryption_at_rest=dict(type='str', choices=['present', 'absent']), set_ntp_info=dict(type='dict', options=dict(broadcastclient=dict(type='bool', default=False), ntp_servers=dict(type='list', elements='str'))), enable_virtual_volumes=dict(type='bool', default=True)))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        if HAS_SF_SDK is False:
            self.module.fail_json(msg='Unable to import the SolidFire Python SDK')
        else:
            self.sfe = netapp_utils.create_sf_connection(module=self.module)

    def get_ntp_details(self):
        """
        get ntp info
        """
        ntp_details = self.sfe.get_ntp_info()
        return ntp_details

    def cmp(self, provided_ntp_servers, existing_ntp_servers):
        return (provided_ntp_servers > existing_ntp_servers) - (provided_ntp_servers < existing_ntp_servers)

    def get_cluster_details(self):
        """
        get cluster info
        """
        cluster_details = self.sfe.get_cluster_info()
        return cluster_details

    def get_vvols_status(self):
        """
        get vvols status
        """
        feature_status = self.sfe.get_feature_status(feature='vvols')
        if feature_status is not None:
            return feature_status.features[0].enabled
        return None

    def get_cluster_full_threshold_status(self):
        """
        get cluster full threshold
        """
        cluster_full_threshold_status = self.sfe.get_cluster_full_threshold()
        return cluster_full_threshold_status

    def setup_ntp_info(self, servers, broadcastclient=None):
        """
        configure ntp
        """
        try:
            self.sfe.set_ntp_info(servers, broadcastclient)
        except Exception as exception_object:
            self.module.fail_json(msg='Error configuring ntp %s' % to_native(exception_object), exception=traceback.format_exc())

    def set_encryption_at_rest(self, state=None):
        """
        enable/disable encryption at rest
        """
        try:
            if state == 'present':
                encryption_state = 'enable'
                self.sfe.enable_encryption_at_rest()
            elif state == 'absent':
                encryption_state = 'disable'
                self.sfe.disable_encryption_at_rest()
        except Exception as exception_object:
            self.module.fail_json(msg='Failed to %s rest encryption %s' % (encryption_state, to_native(exception_object)), exception=traceback.format_exc())

    def enable_feature(self, feature):
        """
        enable feature
        """
        try:
            self.sfe.enable_feature(feature=feature)
        except Exception as exception_object:
            self.module.fail_json(msg='Error enabling %s %s' % (feature, to_native(exception_object)), exception=traceback.format_exc())

    def set_cluster_full_threshold(self, stage2_aware_threshold=None, stage3_block_threshold_percent=None, max_metadata_over_provision_factor=None):
        """
        modify cluster full threshold
        """
        try:
            self.sfe.modify_cluster_full_threshold(stage2_aware_threshold=stage2_aware_threshold, stage3_block_threshold_percent=stage3_block_threshold_percent, max_metadata_over_provision_factor=max_metadata_over_provision_factor)
        except Exception as exception_object:
            self.module.fail_json(msg='Failed to modify cluster full threshold %s' % to_native(exception_object), exception=traceback.format_exc())

    def apply(self):
        """
        Cluster configuration
        """
        changed = False
        result_message = None
        if self.parameters.get('modify_cluster_full_threshold') is not None:
            cluster_full_threshold_details = self.get_cluster_full_threshold_status()
            current_mmopf = cluster_full_threshold_details.max_metadata_over_provision_factor
            current_s3btp = cluster_full_threshold_details.stage3_block_threshold_percent
            current_s2at = cluster_full_threshold_details.stage2_aware_threshold
            if self.parameters.get('modify_cluster_full_threshold')['max_metadata_over_provision_factor'] is not None and current_mmopf != self.parameters['modify_cluster_full_threshold']['max_metadata_over_provision_factor'] or (self.parameters.get('modify_cluster_full_threshold')['stage3_block_threshold_percent'] is not None and current_s3btp != self.parameters['modify_cluster_full_threshold']['stage3_block_threshold_percent']) or (self.parameters.get('modify_cluster_full_threshold')['stage2_aware_threshold'] is not None and current_s2at != self.parameters['modify_cluster_full_threshold']['stage2_aware_threshold']):
                changed = True
                self.set_cluster_full_threshold(self.parameters['modify_cluster_full_threshold']['stage2_aware_threshold'], self.parameters['modify_cluster_full_threshold']['stage3_block_threshold_percent'], self.parameters['modify_cluster_full_threshold']['max_metadata_over_provision_factor'])
        if self.parameters.get('encryption_at_rest') is not None:
            cluster_info = self.get_cluster_details()
            current_encryption_at_rest_state = cluster_info.cluster_info.encryption_at_rest_state
            if current_encryption_at_rest_state == 'disabled' and self.parameters['encryption_at_rest'] == 'present' or (current_encryption_at_rest_state == 'enabled' and self.parameters['encryption_at_rest'] == 'absent'):
                changed = True
                self.set_encryption_at_rest(self.parameters['encryption_at_rest'])
        if self.parameters.get('set_ntp_info') is not None:
            ntp_details = self.get_ntp_details()
            ntp_servers = ntp_details.servers
            broadcast_client = ntp_details.broadcastclient
            if self.parameters.get('set_ntp_info')['broadcastclient'] != broadcast_client or self.cmp(self.parameters.get('set_ntp_info')['ntp_servers'], ntp_servers) != 0:
                changed = True
                self.setup_ntp_info(self.parameters.get('set_ntp_info')['ntp_servers'], self.parameters.get('set_ntp_info')['broadcastclient'])
        if self.parameters.get('enable_virtual_volumes') is not None:
            current_vvols_status = self.get_vvols_status()
            if current_vvols_status is False and self.parameters.get('enable_virtual_volumes') is True:
                changed = True
                self.enable_feature('vvols')
            elif current_vvols_status is True and self.parameters.get('enable_virtual_volumes') is not True:
                self.module.fail_json(msg='Error disabling vvols: this feature cannot be undone')
        if self.module.check_mode is True:
            result_message = 'Check mode, skipping changes'
        self.module.exit_json(changed=changed, msg=result_message)