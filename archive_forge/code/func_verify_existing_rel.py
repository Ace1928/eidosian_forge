from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def verify_existing_rel(self, rel_data):
    if self.existing_rel_data:
        master_volume, aux_volume = (rel_data['master_vdisk_name'], rel_data['aux_vdisk_name'])
        primary, remotecluster, rel_type = (rel_data['primary'], rel_data['aux_cluster_name'], rel_data['copy_type'])
        if rel_type != 'migration':
            self.module.fail_json(msg='Remote Copy relationship [%s] already exists and is not a migration relationship' % self.relationship_name)
        if self.source_volume != master_volume:
            self.module.fail_json(msg='Migration relationship [%s] already exists with a different source volume' % self.relationship_name)
        if self.target_volume != aux_volume:
            self.module.fail_json(msg='Migration relationship [%s] already exists with a different target volume' % self.relationship_name)
        if primary != 'master':
            self.module.fail_json(msg='Migration relationship [%s] replication direction is incorrect' % self.relationship_name)
        if remotecluster != self.remote_cluster:
            self.module.fail_json(msg='Migration relationship [%s] is configured with a different partner system' % self.relationship_name)