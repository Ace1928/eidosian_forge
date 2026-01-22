from __future__ import absolute_import, division, print_function
import codecs
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text, to_bytes
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_owning_resource, rest_vserver
def add_naa_id(self, info):
    """ https://kb.netapp.com/Advice_and_Troubleshooting/Data_Storage_Systems/FlexPod_with_Infrastructure_Automation/
            How_to_match__LUNs_NAA_number_to_its_serial_number
        """
    if info and 'records' in info:
        for lun in info['records']:
            if 'serial_number' in lun:
                hexlify = codecs.getencoder('hex')
                lun['serial_hex'] = to_text(hexlify(to_bytes(lun['serial_number']))[0])
                lun['naa_id'] = 'naa.600a0980' + lun['serial_hex']