import copy
import hmac
import uuid
import base64
import datetime
from hashlib import sha1
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import ET, b, httplib, urlencode
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.common.aws import AWSGenericResponse, AWSTokenConnection
from libcloud.common.base import ConnectionUserAndKey
from libcloud.common.types import LibcloudError
def _update_multi_value_record(self, record, name=None, type=None, data=None, extra=None):
    other_records = record.extra.get('_other_records', [])
    attrs = {'xmlns': NAMESPACE}
    changeset = ET.Element('ChangeResourceRecordSetsRequest', attrs)
    batch = ET.SubElement(changeset, 'ChangeBatch')
    changes = ET.SubElement(batch, 'Changes')
    change = ET.SubElement(changes, 'Change')
    ET.SubElement(change, 'Action').text = 'DELETE'
    rrs = ET.SubElement(change, 'ResourceRecordSet')
    if record.name:
        record_name = record.name + '.' + record.zone.domain
    else:
        record_name = record.zone.domain
    ET.SubElement(rrs, 'Name').text = record_name
    ET.SubElement(rrs, 'Type').text = self.RECORD_TYPE_MAP[record.type]
    ET.SubElement(rrs, 'TTL').text = str(record.extra.get('ttl', '0'))
    rrecs = ET.SubElement(rrs, 'ResourceRecords')
    rrec = ET.SubElement(rrecs, 'ResourceRecord')
    ET.SubElement(rrec, 'Value').text = record.data
    for other_record in other_records:
        rrec = ET.SubElement(rrecs, 'ResourceRecord')
        ET.SubElement(rrec, 'Value').text = other_record['data']
    change = ET.SubElement(changes, 'Change')
    ET.SubElement(change, 'Action').text = 'CREATE'
    rrs = ET.SubElement(change, 'ResourceRecordSet')
    if name:
        record_name = name + '.' + record.zone.domain
    else:
        record_name = record.zone.domain
    ET.SubElement(rrs, 'Name').text = record_name
    ET.SubElement(rrs, 'Type').text = self.RECORD_TYPE_MAP[type]
    ET.SubElement(rrs, 'TTL').text = str(extra.get('ttl', '0'))
    rrecs = ET.SubElement(rrs, 'ResourceRecords')
    rrec = ET.SubElement(rrecs, 'ResourceRecord')
    ET.SubElement(rrec, 'Value').text = data
    for other_record in other_records:
        rrec = ET.SubElement(rrecs, 'ResourceRecord')
        ET.SubElement(rrec, 'Value').text = other_record['data']
    uri = API_ROOT + 'hostedzone/' + record.zone.id + '/rrset'
    data = ET.tostring(changeset)
    self.connection.set_context({'zone_id': record.zone.id})
    response = self.connection.request(uri, method='POST', data=data)
    return response.status == httplib.OK