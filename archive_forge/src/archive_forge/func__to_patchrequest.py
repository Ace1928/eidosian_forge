import re
import json
import hashlib
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError, MalformedResponseError
from libcloud.common.exceptions import BaseHTTPError
def _to_patchrequest(self, zone, record, name, type, data, extra, action):
    rrset = {}
    cur_records = self.list_records(Zone(id=zone, domain=None, type=None, ttl=None, driver=self))
    if name != '':
        rrset['name'] = name + '.' + zone + '.'
    else:
        rrset['name'] = zone + '.'
    rrset['type'] = type
    rrset['changetype'] = action
    rrset['records'] = []
    if not (extra is None or extra.get('ttl', None) is None):
        rrset['ttl'] = extra['ttl']
    content = {}
    if not action == 'delete':
        content['content'] = data
        if not (extra is None or extra.get('disabled', None) is None):
            content['disabled'] = extra['disabled']
        rrset['records'].append(content)
    id = hashlib.md5(str(name + ' ' + data).encode('utf-8')).hexdigest()
    for r in cur_records:
        if action == 'update' and r.id == record.id:
            continue
        if name == r.name and r.id != id:
            rrset['changetype'] = 'update'
            content = {}
            content['content'] = r.data
            if not (r.extra is None or r.extra.get('disabled', None) is None):
                content['disabled'] = r.extra['disabled']
            rrset['records'].append(content)
    request = list()
    request.append(rrset)
    return request