from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
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