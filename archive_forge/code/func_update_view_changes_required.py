from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def update_view_changes_required(self):
    """Determine the changes required for snapshot consistency group point-in-time view."""
    changes = {'expand_reserve_capacity': [], 'trim_reserve_capacity': [], 'map_snapshot_volumes_mapping': [], 'unmap_snapshot_volumes_mapping': [], 'move_snapshot_volumes_mapping': [], 'update_snapshot_volumes_writable': []}
    view = self.get_consistency_group_view()
    host_objects_by_name = self.get_all_hosts_and_hostgroups_by_name()
    host_objects_by_id = self.get_all_hosts_and_hostgroups_by_id()
    existing_volumes_by_id = self.get_all_volumes_by_id()
    if view:
        if len(view['snapshot_volumes']) != len(self.volumes):
            self.module.fail_json(msg='Cannot add or remove snapshot volumes once view is created! Group [%s]. Array [%s].' % (self.group_name, self.ssid))
        expand_volumes = {}
        writable_volumes = {}
        for snapshot_volume in view['snapshot_volumes']:
            for volume_name, volume_info in self.volumes.items():
                if existing_volumes_by_id[snapshot_volume['baseVol']]['name'] == volume_name:
                    if volume_info['snapshot_volume_host'] and (not snapshot_volume['listOfMappings']):
                        changes['map_snapshot_volumes_mapping'].append({'mappableObjectId': snapshot_volume['id'], 'lun': volume_info['snapshot_volume_lun'], 'targetId': host_objects_by_name[volume_info['snapshot_volume_host']]['id']})
                    elif not volume_info['snapshot_volume_host'] and snapshot_volume['listOfMappings']:
                        changes['unmap_snapshot_volumes_mapping'].append({'snapshot_volume_name': snapshot_volume['name'], 'lun_mapping_reference': snapshot_volume['listOfMappings'][0]['lunMappingRef']})
                    elif snapshot_volume['listOfMappings'] and (volume_info['snapshot_volume_host'] != host_objects_by_id[snapshot_volume['listOfMappings'][0]['mapRef']]['name'] or volume_info['snapshot_volume_lun'] != snapshot_volume['listOfMappings'][0]['lun']):
                        changes['move_snapshot_volumes_mapping'].append({'lunMappingRef': snapshot_volume['listOfMappings'][0]['lunMappingRef'], 'lun': volume_info['snapshot_volume_lun'], 'mapRef': host_objects_by_name[volume_info['snapshot_volume_host']]['id']})
                    if volume_info['snapshot_volume_writable'] != (snapshot_volume['accessMode'] == 'readWrite'):
                        volume_info.update({'snapshot_volume_id': snapshot_volume['id']})
                        writable_volumes.update({volume_name: volume_info})
                    if volume_info['snapshot_volume_writable'] and snapshot_volume['accessMode'] == 'readWrite':
                        current_reserve_capacity_pct = int(round(float(snapshot_volume['repositoryCapacity']) / float(snapshot_volume['baseVolumeCapacity']) * 100))
                        if volume_info['reserve_capacity_pct'] > current_reserve_capacity_pct:
                            expand_reserve_capacity_pct = volume_info['reserve_capacity_pct'] - current_reserve_capacity_pct
                            expand_volumes.update({volume_name: {'reserve_capacity_pct': expand_reserve_capacity_pct, 'preferred_reserve_storage_pool': volume_info['preferred_reserve_storage_pool'], 'reserve_volume_id': snapshot_volume['repositoryVolume']}})
                        elif volume_info['reserve_capacity_pct'] < current_reserve_capacity_pct:
                            existing_volumes_by_id = self.get_all_volumes_by_id()
                            existing_volumes_by_name = self.get_all_volumes_by_name()
                            existing_concat_volumes_by_id = self.get_all_concat_volumes_by_id()
                            trim_pct = current_reserve_capacity_pct - volume_info['reserve_capacity_pct']
                            concat_volume_id = snapshot_volume['repositoryVolume']
                            concat_volume_info = existing_concat_volumes_by_id[concat_volume_id]
                            base_volume_info = existing_volumes_by_name[volume_name]
                            base_volume_size_bytes = int(base_volume_info['totalSizeInBytes'])
                            total_member_volume_size_bytes = 0
                            member_volumes_to_trim = []
                            for trim_count, member_volume_id in enumerate(reversed(concat_volume_info['memberRefs'][1:])):
                                member_volume_info = existing_volumes_by_id[member_volume_id]
                                member_volumes_to_trim.append(member_volume_info)
                                total_member_volume_size_bytes += int(member_volume_info['totalSizeInBytes'])
                                total_trimmed_size_pct = round(total_member_volume_size_bytes / base_volume_size_bytes * 100)
                                if total_trimmed_size_pct >= trim_pct:
                                    changes['trim_reserve_capacity'].append({'concat_volume_id': concat_volume_id, 'trim_count': trim_count + 1})
                                    if total_trimmed_size_pct > trim_pct:
                                        expand_reserve_capacity_pct = total_trimmed_size_pct - trim_pct
                                        expand_volumes.update({volume_name: {'reserve_capacity_pct': expand_reserve_capacity_pct, 'preferred_reserve_storage_pool': volume_info['preferred_reserve_storage_pool'], 'reserve_volume_id': snapshot_volume['repositoryVolume']}})
                                    break
                            else:
                                initial_reserve_volume_info = existing_volumes_by_id[concat_volume_info['memberRefs'][0]]
                                minimum_capacity_pct = round(int(initial_reserve_volume_info['totalSizeInBytes']) / base_volume_size_bytes * 100)
                                self.module.fail_json(msg='Cannot delete initial reserve capacity volume! Minimum reserve capacity percent [%s]. Base volume [%s]. Group [%s]. Array [%s].' % (minimum_capacity_pct, volume_name, self.group_name, self.ssid))
        changes.update({'expand_reserve_capacity': expand_volumes, 'update_snapshot_volumes_writable': writable_volumes})
    return changes