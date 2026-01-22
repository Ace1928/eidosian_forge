from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
class AnsibleCloudStackVolume(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackVolume, self).__init__(module)
        self.returns = {'group': 'group', 'attached': 'attached', 'vmname': 'vm', 'deviceid': 'device_id', 'type': 'type', 'size': 'size', 'url': 'url'}
        self.volume = None

    def get_volume(self):
        if not self.volume:
            args = {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'zoneid': self.get_zone(key='id'), 'displayvolume': self.module.params.get('display_volume'), 'type': 'DATADISK', 'fetch_list': True}
            if self.module.params.get('state') == 'extracted':
                del args['type']
            volumes = self.query_api('listVolumes', **args)
            if volumes:
                volume_name = self.module.params.get('name')
                for v in volumes:
                    if volume_name.lower() == v['name'].lower():
                        self.volume = v
                        break
        return self.volume

    def get_snapshot(self, key=None):
        snapshot = self.module.params.get('snapshot')
        if not snapshot:
            return None
        args = {'name': snapshot, 'account': self.get_account('name'), 'domainid': self.get_domain('id'), 'projectid': self.get_project('id')}
        snapshots = self.query_api('listSnapshots', **args)
        if snapshots:
            return self._get_by_key(key, snapshots['snapshot'][0])
        self.module.fail_json(msg='Snapshot with name %s not found' % snapshot)

    def present_volume(self):
        volume = self.get_volume()
        if volume:
            volume = self.update_volume(volume)
        else:
            disk_offering_id = self.get_disk_offering(key='id')
            snapshot_id = self.get_snapshot(key='id')
            if not disk_offering_id and (not snapshot_id):
                self.module.fail_json(msg='Required one of: disk_offering,snapshot')
            self.result['changed'] = True
            args = {'name': self.module.params.get('name'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'diskofferingid': disk_offering_id, 'displayvolume': self.module.params.get('display_volume'), 'maxiops': self.module.params.get('max_iops'), 'miniops': self.module.params.get('min_iops'), 'projectid': self.get_project(key='id'), 'size': self.module.params.get('size'), 'snapshotid': snapshot_id, 'zoneid': self.get_zone(key='id')}
            if not self.module.check_mode:
                res = self.query_api('createVolume', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    volume = self.poll_job(res, 'volume')
        if volume:
            volume = self.ensure_tags(resource=volume, resource_type='Volume')
            self.volume = volume
        return volume

    def attached_volume(self):
        volume = self.present_volume()
        if volume:
            if volume.get('virtualmachineid') != self.get_vm(key='id'):
                self.result['changed'] = True
                if not self.module.check_mode:
                    volume = self.detached_volume()
            if 'attached' not in volume:
                self.result['changed'] = True
                args = {'id': volume['id'], 'virtualmachineid': self.get_vm(key='id'), 'deviceid': self.module.params.get('device_id')}
                if not self.module.check_mode:
                    res = self.query_api('attachVolume', **args)
                    poll_async = self.module.params.get('poll_async')
                    if poll_async:
                        volume = self.poll_job(res, 'volume')
        return volume

    def detached_volume(self):
        volume = self.present_volume()
        if volume:
            if 'attached' not in volume:
                return volume
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('detachVolume', id=volume['id'])
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    volume = self.poll_job(res, 'volume')
        return volume

    def absent_volume(self):
        volume = self.get_volume()
        if volume:
            if 'attached' in volume and (not self.module.params.get('force')):
                self.module.fail_json(msg="Volume '%s' is attached, use force=true for detaching and removing the volume." % volume.get('name'))
            self.result['changed'] = True
            if not self.module.check_mode:
                volume = self.detached_volume()
                res = self.query_api('deleteVolume', id=volume['id'])
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    self.poll_job(res, 'volume')
        return volume

    def update_volume(self, volume):
        args_resize = {'id': volume['id'], 'diskofferingid': self.get_disk_offering(key='id'), 'maxiops': self.module.params.get('max_iops'), 'miniops': self.module.params.get('min_iops'), 'size': self.module.params.get('size')}
        volume_copy = volume.copy()
        volume_copy['size'] = volume_copy['size'] / 2 ** 30
        if self.has_changed(args_resize, volume_copy):
            self.result['changed'] = True
            if not self.module.check_mode:
                args_resize['shrinkok'] = self.module.params.get('shrink_ok')
                res = self.query_api('resizeVolume', **args_resize)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    volume = self.poll_job(res, 'volume')
                self.volume = volume
        return volume

    def extract_volume(self):
        volume = self.get_volume()
        if not volume:
            self.module.fail_json(msg='Failed: volume not found')
        args = {'id': volume['id'], 'url': self.module.params.get('url'), 'mode': self.module.params.get('mode').upper(), 'zoneid': self.get_zone(key='id')}
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('extractVolume', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                volume = self.poll_job(res, 'volume')
            self.volume = volume
        return volume

    def upload_volume(self):
        volume = self.get_volume()
        if not volume:
            disk_offering_id = self.get_disk_offering(key='id')
            self.result['changed'] = True
            args = {'name': self.module.params.get('name'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'zoneid': self.get_zone(key='id'), 'format': self.module.params.get('format'), 'url': self.module.params.get('url'), 'diskofferingid': disk_offering_id}
            if not self.module.check_mode:
                res = self.query_api('uploadVolume', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    volume = self.poll_job(res, 'volume')
        if volume:
            volume = self.ensure_tags(resource=volume, resource_type='Volume')
            self.volume = volume
        return volume