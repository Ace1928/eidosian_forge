import copy
import base64
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import ET, b, httplib
from libcloud.utils.xml import findall, findtext
from libcloud.utils.misc import get_new_obj, merge_valid_keys
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
def _to_record_elem(self, name=None, type=None, data=None, extra=None):
    record_elem = ET.Element('host', {})
    if name:
        name_elem = ET.SubElement(record_elem, 'hostname')
        name_elem.text = name
    if type is not None:
        type_elem = ET.SubElement(record_elem, 'host-type')
        type_elem.text = self.RECORD_TYPE_MAP[type]
    if data:
        data_elem = ET.SubElement(record_elem, 'data')
        data_elem.text = data
    if extra:
        if 'ttl' in extra:
            ttl_elem = ET.SubElement(record_elem, 'ttl', {'type': 'integer'})
            ttl_elem.text = str(extra['ttl'])
        if 'priority' in extra:
            priority_elem = ET.SubElement(record_elem, 'priority', {'type': 'integer'})
            priority_elem.text = str(extra['priority'])
        if 'notes' in extra:
            notes_elem = ET.SubElement(record_elem, 'notes')
            notes_elem.text = extra['notes']
    return record_elem