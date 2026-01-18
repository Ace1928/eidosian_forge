from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
import copy
def unmap_volume_from_sdc(self, volume, sdc):
    """Unmap SDC's from volume
            :param volume: volume details
            :param sdc: List of SDCs to be unmapped
            :return: Boolean indicating if unmap operation is successful
        """
    current_sdcs = volume['mappedSdcInfo']
    current_sdc_ids = []
    sdc_id_list = []
    sdc_id = None
    if current_sdcs:
        for temp in current_sdcs:
            current_sdc_ids.append(temp['sdcId'])
    for temp in sdc:
        if 'sdc_name' in temp and temp['sdc_name']:
            sdc_id = self.get_sdc_id(sdc_name=temp['sdc_name'])
        elif 'sdc_ip' in temp and temp['sdc_ip']:
            sdc_id = self.get_sdc_id(sdc_ip=temp['sdc_ip'])
        else:
            sdc_id = self.get_sdc_id(sdc_id=temp['sdc_id'])
        if sdc_id in current_sdc_ids:
            sdc_id_list.append(sdc_id)
    LOG.info('SDC IDs to remove %s', sdc_id_list)
    if len(sdc_id_list) == 0:
        return False
    try:
        for sdc_id in sdc_id_list:
            self.powerflex_conn.volume.remove_mapped_sdc(volume['id'], sdc_id)
        return True
    except Exception as e:
        errormsg = 'Unmap SDC {0} from volume {1} failed with error {2}'.format(sdc_id, volume['id'], str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)