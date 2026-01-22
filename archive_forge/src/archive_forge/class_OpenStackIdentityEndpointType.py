import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackIdentityEndpointType:
    """
    Enum class for openstack identity endpoint type.
    """
    INTERNAL = 'internal'
    EXTERNAL = 'external'
    ADMIN = 'admin'