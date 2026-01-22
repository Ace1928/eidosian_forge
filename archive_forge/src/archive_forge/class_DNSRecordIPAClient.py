from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
class DNSRecordIPAClient(IPAClient):

    def __init__(self, module, host, port, protocol):
        super(DNSRecordIPAClient, self).__init__(module, host, port, protocol)

    def dnsrecord_find(self, zone_name, record_name):
        if record_name == '@':
            return self._post_json(method='dnsrecord_show', name=zone_name, item={'idnsname': record_name, 'all': True})
        else:
            return self._post_json(method='dnsrecord_find', name=zone_name, item={'idnsname': record_name, 'all': True})

    def dnsrecord_add(self, zone_name=None, record_name=None, details=None):
        item = dict(idnsname=record_name)
        if details.get('record_ttl'):
            item.update(dnsttl=details['record_ttl'])
        for value in details['record_values']:
            if details['record_type'] == 'A':
                item.update(a_part_ip_address=value)
            elif details['record_type'] == 'AAAA':
                item.update(aaaa_part_ip_address=value)
            elif details['record_type'] == 'A6':
                item.update(a6_part_data=value)
            elif details['record_type'] == 'CNAME':
                item.update(cname_part_hostname=value)
            elif details['record_type'] == 'DNAME':
                item.update(dname_part_target=value)
            elif details['record_type'] == 'NS':
                item.update(ns_part_hostname=value)
            elif details['record_type'] == 'PTR':
                item.update(ptr_part_hostname=value)
            elif details['record_type'] == 'TXT':
                item.update(txtrecord=value)
            elif details['record_type'] == 'SRV':
                item.update(srvrecord=value)
            elif details['record_type'] == 'MX':
                item.update(mxrecord=value)
            self._post_json(method='dnsrecord_add', name=zone_name, item=item)

    def dnsrecord_mod(self, zone_name=None, record_name=None, details=None):
        item = get_dnsrecord_dict(details)
        item.update(idnsname=record_name)
        if details.get('record_ttl'):
            item.update(dnsttl=details['record_ttl'])
        return self._post_json(method='dnsrecord_mod', name=zone_name, item=item)

    def dnsrecord_del(self, zone_name=None, record_name=None, details=None):
        item = get_dnsrecord_dict(details)
        item.update(idnsname=record_name)
        return self._post_json(method='dnsrecord_del', name=zone_name, item=item)