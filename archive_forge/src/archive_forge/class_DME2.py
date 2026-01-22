from __future__ import absolute_import, division, print_function
import json
import hashlib
import hmac
import locale
from time import strftime, gmtime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.six import string_types
class DME2(object):

    def __init__(self, apikey, secret, domain, sandbox, module):
        self.module = module
        self.api = apikey
        self.secret = secret
        if sandbox:
            self.baseurl = 'https://api.sandbox.dnsmadeeasy.com/V2.0/'
            self.module.warn(warning='Sandbox is enabled. All actions are made against the URL %s' % self.baseurl)
        else:
            self.baseurl = 'https://api.dnsmadeeasy.com/V2.0/'
        self.domain = str(domain)
        self.domain_map = None
        self.record_map = None
        self.records = None
        self.all_records = None
        self.contactList_map = None
        if not self.domain.isdigit():
            self.domain = self.getDomainByName(self.domain)['id']
        self.record_url = 'dns/managed/' + str(self.domain) + '/records'
        self.monitor_url = 'monitor'
        self.contactList_url = 'contactList'

    def _headers(self):
        currTime = self._get_date()
        hashstring = self._create_hash(currTime)
        headers = {'x-dnsme-apiKey': self.api, 'x-dnsme-hmac': hashstring, 'x-dnsme-requestDate': currTime, 'content-type': 'application/json'}
        return headers

    def _get_date(self):
        locale.setlocale(locale.LC_TIME, 'C')
        return strftime('%a, %d %b %Y %H:%M:%S GMT', gmtime())

    def _create_hash(self, rightnow):
        return hmac.new(self.secret.encode(), rightnow.encode(), hashlib.sha1).hexdigest()

    def query(self, resource, method, data=None):
        url = self.baseurl + resource
        if data and (not isinstance(data, string_types)):
            data = urlencode(data)
        response, info = fetch_url(self.module, url, data=data, method=method, headers=self._headers())
        if info['status'] not in (200, 201, 204):
            self.module.fail_json(msg='%s returned %s, with body: %s' % (url, info['status'], info['msg']))
        try:
            return json.load(response)
        except Exception:
            return {}

    def getDomain(self, domain_id):
        if not self.domain_map:
            self._instMap('domain')
        return self.domains.get(domain_id, False)

    def getDomainByName(self, domain_name):
        if not self.domain_map:
            self._instMap('domain')
        return self.getDomain(self.domain_map.get(domain_name, 0))

    def getDomains(self):
        return self.query('dns/managed', 'GET')['data']

    def getRecord(self, record_id):
        if not self.record_map:
            self._instMap('record')
        return self.records.get(record_id, False)

    def getMatchingRecord(self, record_name, record_type, record_value):
        if not self.all_records:
            self.all_records = self.getRecords()
        if record_type in ['CNAME', 'ANAME', 'HTTPRED', 'PTR']:
            for result in self.all_records:
                if result['name'] == record_name and result['type'] == record_type:
                    return result
            return False
        elif record_type in ['A', 'AAAA', 'MX', 'NS', 'TXT', 'SRV']:
            for result in self.all_records:
                if record_type == 'MX':
                    value = record_value.split(' ')[1]
                elif record_type == 'TXT':
                    value = '"{0}"'.format(record_value)
                elif record_type == 'SRV':
                    value = record_value.split(' ')[3]
                else:
                    value = record_value
                if result['name'] == record_name and result['type'] == record_type and (result['value'] == value):
                    return result
            return False
        else:
            raise Exception('record_type not yet supported')

    def getRecords(self):
        return self.query(self.record_url, 'GET')['data']

    def _instMap(self, type):
        map = {}
        results = {}
        for result in getattr(self, 'get' + type.title() + 's')():
            map[result['name']] = result['id']
            results[result['id']] = result
        setattr(self, type + '_map', map)
        setattr(self, type + 's', results)

    def prepareRecord(self, data):
        return json.dumps(data, separators=(',', ':'))

    def createRecord(self, data):
        return self.query(self.record_url, 'POST', data)

    def updateRecord(self, record_id, data):
        return self.query(self.record_url + '/' + str(record_id), 'PUT', data)

    def deleteRecord(self, record_id):
        return self.query(self.record_url + '/' + str(record_id), 'DELETE')

    def getMonitor(self, record_id):
        return self.query(self.monitor_url + '/' + str(record_id), 'GET')

    def updateMonitor(self, record_id, data):
        return self.query(self.monitor_url + '/' + str(record_id), 'PUT', data)

    def prepareMonitor(self, data):
        return json.dumps(data, separators=(',', ':'))

    def getContactList(self, contact_list_id):
        if not self.contactList_map:
            self._instMap('contactList')
        return self.contactLists.get(contact_list_id, False)

    def getContactlists(self):
        return self.query(self.contactList_url, 'GET')['data']

    def getContactListByName(self, name):
        if not self.contactList_map:
            self._instMap('contactList')
        return self.getContactList(self.contactList_map.get(name, 0))