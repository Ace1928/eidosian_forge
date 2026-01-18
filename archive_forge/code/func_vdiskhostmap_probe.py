from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def vdiskhostmap_probe(self, mdata):
    props = []
    self.log("vdiskhostmap_probe props='%s'", mdata)
    mapping_exist = False
    for data in mdata:
        if self.host:
            if self.host == data['host_name'] and self.volname == data['name']:
                if self.scsi and self.scsi != int(data['SCSI_id']):
                    self.module.fail_json(msg='Update not supported for parameter: scsi')
                mapping_exist = True
        elif self.hostcluster:
            if self.hostcluster == data['host_cluster_name'] and self.volname == data['name']:
                if self.scsi and self.scsi != int(data['SCSI_id']):
                    self.module.fail_json(msg='Update not supported for parameter: scsi')
                mapping_exist = True
    if not mapping_exist:
        props += ['map']
    if props is []:
        props = None
    self.log("vdiskhostmap_probe props='%s'", props)
    return props