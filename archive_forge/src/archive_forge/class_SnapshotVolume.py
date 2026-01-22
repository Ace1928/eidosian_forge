from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
class SnapshotVolume(object):

    def __init__(self):
        argument_spec = basic_auth_argument_spec()
        argument_spec.update(dict(api_username=dict(type='str', required=True), api_password=dict(type='str', required=True, no_log=True), api_url=dict(type='str', required=True), ssid=dict(type='str', required=True), snapshot_image_id=dict(type='str', required=True), full_threshold=dict(type='int', default=85), name=dict(type='str', required=True), view_mode=dict(type='str', default='readOnly', choices=['readOnly', 'readWrite', 'modeUnknown', '__Undefined']), repo_percentage=dict(type='int', default=20), storage_pool_name=dict(type='str', required=True), state=dict(type='str', required=True, choices=['absent', 'present'])))
        self.module = AnsibleModule(argument_spec=argument_spec)
        args = self.module.params
        self.state = args['state']
        self.ssid = args['ssid']
        self.snapshot_image_id = args['snapshot_image_id']
        self.full_threshold = args['full_threshold']
        self.name = args['name']
        self.view_mode = args['view_mode']
        self.repo_percentage = args['repo_percentage']
        self.storage_pool_name = args['storage_pool_name']
        self.url = args['api_url']
        self.user = args['api_username']
        self.pwd = args['api_password']
        self.certs = args['validate_certs']
        if not self.url.endswith('/'):
            self.url += '/'

    @property
    def pool_id(self):
        pools = 'storage-systems/%s/storage-pools' % self.ssid
        url = self.url + pools
        rc, data = request(url, headers=HEADERS, url_username=self.user, url_password=self.pwd, validate_certs=self.certs)
        for pool in data:
            if pool['name'] == self.storage_pool_name:
                self.pool_data = pool
                return pool['id']
        self.module.fail_json(msg="No storage pool with the name: '%s' was found" % self.name)

    @property
    def ss_vol_exists(self):
        rc, ss_vols = request(self.url + 'storage-systems/%s/snapshot-volumes' % self.ssid, headers=HEADERS, url_username=self.user, url_password=self.pwd, validate_certs=self.certs)
        if ss_vols:
            for ss_vol in ss_vols:
                if ss_vol['name'] == self.name:
                    self.ss_vol = ss_vol
                    return True
        else:
            return False
        return False

    @property
    def ss_vol_needs_update(self):
        if self.ss_vol['fullWarnThreshold'] != self.full_threshold:
            return True
        else:
            return False

    def create_ss_vol(self):
        post_data = dict(snapshotImageId=self.snapshot_image_id, fullThreshold=self.full_threshold, name=self.name, viewMode=self.view_mode, repositoryPercentage=self.repo_percentage, repositoryPoolId=self.pool_id)
        rc, create_resp = request(self.url + 'storage-systems/%s/snapshot-volumes' % self.ssid, data=json.dumps(post_data), headers=HEADERS, url_username=self.user, url_password=self.pwd, validate_certs=self.certs, method='POST')
        self.ss_vol = create_resp
        if self.ss_vol_needs_update:
            self.update_ss_vol()
        else:
            self.module.exit_json(changed=True, **create_resp)

    def update_ss_vol(self):
        post_data = dict(fullThreshold=self.full_threshold)
        rc, resp = request(self.url + 'storage-systems/%s/snapshot-volumes/%s' % (self.ssid, self.ss_vol['id']), data=json.dumps(post_data), headers=HEADERS, url_username=self.user, url_password=self.pwd, method='POST', validate_certs=self.certs)
        self.module.exit_json(changed=True, **resp)

    def remove_ss_vol(self):
        rc, resp = request(self.url + 'storage-systems/%s/snapshot-volumes/%s' % (self.ssid, self.ss_vol['id']), headers=HEADERS, url_username=self.user, url_password=self.pwd, validate_certs=self.certs, method='DELETE')
        self.module.exit_json(changed=True, msg='Volume successfully deleted')

    def apply(self):
        if self.state == 'present':
            if self.ss_vol_exists:
                if self.ss_vol_needs_update:
                    self.update_ss_vol()
                else:
                    self.module.exit_json(changed=False, **self.ss_vol)
            else:
                self.create_ss_vol()
        elif self.ss_vol_exists:
            self.remove_ss_vol()
        else:
            self.module.exit_json(changed=False, msg='Volume already absent')