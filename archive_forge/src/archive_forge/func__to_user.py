import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def _to_user(self, data):
    user = OpenStackIdentityUser(id=data['id'], domain_id=data['domain_id'], name=data['name'], email=data.get('email'), description=data.get('description', None), enabled=data.get('enabled'))
    return user