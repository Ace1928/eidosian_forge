import copy
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import get_new_obj, merge_valid_keys
from libcloud.common.base import PollingConnection
from libcloud.common.types import LibcloudError
from libcloud.common.openstack import OpenStackDriverMixin
from libcloud.common.rackspace import AUTH_URL
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.drivers.openstack import OpenStack_1_1_Response, OpenStack_1_1_Connection
def _to_ptr_record(self, data, link):
    id = data['id']
    ip = data['data']
    domain = data['name']
    extra = {'uri': link['href'], 'service_name': link['rel']}
    for key in VALID_RECORD_EXTRA_PARAMS:
        if key in data:
            extra[key] = data[key]
    record = RackspacePTRRecord(id=str(id), ip=ip, domain=domain, driver=self, extra=extra)
    return record