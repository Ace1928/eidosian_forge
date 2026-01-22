from __future__ import absolute_import, division, print_function
import xml.etree.ElementTree as ET
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
class VMwareHostLogbundle(PyVmomi):

    def __init__(self, module):
        super(VMwareHostLogbundle, self).__init__(module)
        self.esxi_hostname = self.params['esxi_hostname']
        self.dest = self.params['dest']
        self.manifests = self.params['manifests']
        self.performance_data = self.params['performance_data']
        if not self.dest.endswith('.tgz'):
            self.dest = self.dest + '.tgz'

    def generate_req_headers(self, url):
        req = vim.SessionManager.HttpServiceRequestSpec(method='httpGet', url=url)
        ticket = self.content.sessionManager.AcquireGenericServiceTicket(req)
        headers = {'Content-Type': 'application/octet-stream', 'Cookie': 'vmware_cgi_ticket=%s' % ticket.id}
        return headers

    def validate_manifests(self):
        url = 'https://' + self.esxi_hostname + '/cgi-bin/vm-support.cgi?listmanifests=1'
        headers = self.generate_req_headers(url)
        manifests = []
        try:
            resp, info = fetch_url(self.module, method='GET', headers=headers, url=url)
            if info['status'] != 200:
                self.module.fail_json(msg='failed to fetch manifests from %s: %s' % (url, info['msg']))
            manifest_list = ET.fromstring(resp.read())
            for manifest in manifest_list[0]:
                manifests.append(manifest.attrib['id'])
        except Exception as e:
            self.module.fail_json(msg='Failed to fetch manifests from %s: %s' % (url, e))
        for manifest in self.manifests:
            validate_manifest_result = [m for m in manifests if m == manifest]
            if not validate_manifest_result:
                self.module.fail_json(msg='%s is a manifest that cannot be specified.' % manifest)

    def get_logbundle(self):
        self.validate_manifests()
        url = 'https://' + self.esxi_hostname + '/cgi-bin/vm-support.cgi?manifests=' + '&'.join(self.manifests)
        if self.performance_data:
            duration = self.performance_data.get('duration')
            interval = self.performance_data.get('interval')
            url = url + '&performance=true&duration=%s&interval=%s' % (duration, interval)
        headers = self.generate_req_headers(url)
        try:
            resp, info = fetch_url(self.module, method='GET', headers=headers, url=url)
            if info['status'] != 200:
                self.module.fail_json(msg='failed to fetch logbundle from %s: %s' % (url, info['msg']))
            with open(self.dest, 'wb') as local_file:
                local_file.write(resp.read())
        except Exception as e:
            self.module.fail_json(msg='Failed to fetch logbundle from %s: %s' % (url, e))
        self.module.exit_json(changed=True, dest=self.dest)