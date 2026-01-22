from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackIso(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackIso, self).__init__(module)
        self.returns = {'checksum': 'checksum', 'status': 'status', 'isready': 'is_ready', 'crossZones': 'cross_zones', 'format': 'format', 'ostypename': 'os_type', 'isfeatured': 'is_featured', 'bootable': 'bootable', 'ispublic': 'is_public'}
        self.iso = None

    def _get_common_args(self):
        return {'name': self.module.params.get('name'), 'displaytext': self.get_or_fallback('display_text', 'name'), 'isdynamicallyscalable': self.module.params.get('is_dynamically_scalable'), 'ostypeid': self.get_os_type('id'), 'bootable': self.module.params.get('bootable')}

    def register_iso(self):
        args = self._get_common_args()
        args.update({'domainid': self.get_domain('id'), 'account': self.get_account('name'), 'projectid': self.get_project('id'), 'checksum': self.module.params.get('checksum'), 'isfeatured': self.module.params.get('is_featured'), 'ispublic': self.module.params.get('is_public')})
        if not self.module.params.get('cross_zones'):
            args['zoneid'] = self.get_zone(key='id')
        else:
            args['zoneid'] = -1
        if args['bootable'] and (not args['ostypeid']):
            self.module.fail_json(msg="OS type 'os_type' is required if 'bootable=true'.")
        args['url'] = self.module.params.get('url')
        if not args['url']:
            self.module.fail_json(msg='URL is required.')
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('registerIso', **args)
            self.iso = res['iso'][0]
        return self.iso

    def present_iso(self):
        iso = self.get_iso()
        if not iso:
            iso = self.register_iso()
        else:
            iso = self.update_iso(iso)
        if iso:
            iso = self.ensure_tags(resource=iso, resource_type='ISO')
            self.iso = iso
        return iso

    def update_iso(self, iso):
        args = self._get_common_args()
        args.update({'id': iso['id']})
        if self.has_changed(args, iso):
            self.result['changed'] = True
            if not self.module.params.get('cross_zones'):
                args['zoneid'] = self.get_zone(key='id')
            else:
                self.result['cross_zones'] = True
                args['zoneid'] = -1
            if not self.module.check_mode:
                res = self.query_api('updateIso', **args)
                self.iso = res['iso']
        return self.iso

    def get_iso(self):
        if not self.iso:
            args = {'isready': self.module.params.get('is_ready'), 'isofilter': self.module.params.get('iso_filter'), 'domainid': self.get_domain('id'), 'account': self.get_account('name'), 'projectid': self.get_project('id')}
            if not self.module.params.get('cross_zones'):
                args['zoneid'] = self.get_zone(key='id')
            checksum = self.module.params.get('checksum')
            if not checksum:
                args['name'] = self.module.params.get('name')
            isos = self.query_api('listIsos', **args)
            if isos:
                if not checksum:
                    self.iso = isos['iso'][0]
                else:
                    for i in isos['iso']:
                        if i['checksum'] == checksum:
                            self.iso = i
                            break
        return self.iso

    def absent_iso(self):
        iso = self.get_iso()
        if iso:
            self.result['changed'] = True
            args = {'id': iso['id'], 'projectid': self.get_project('id')}
            if not self.module.params.get('cross_zones'):
                args['zoneid'] = self.get_zone(key='id')
            if not self.module.check_mode:
                res = self.query_api('deleteIso', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    self.poll_job(res, 'iso')
        return iso

    def get_result(self, resource):
        super(AnsibleCloudStackIso, self).get_result(resource)
        if self.module.params.get('cross_zones'):
            self.result['cross_zones'] = True
            if 'zone' in self.result:
                del self.result['zone']
        return self.result