from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataPublicIpBlock:
    """
    DimensionData Public IP Block with location.
    """

    def __init__(self, id, base_ip, size, location, network_domain, status):
        self.id = str(id)
        self.base_ip = base_ip
        self.size = size
        self.location = location
        self.network_domain = network_domain
        self.status = status

    def __repr__(self):
        return '<DimensionDataNetworkDomain: id=%s, base_ip=%s, size=%s, location=%s, status=%s>' % (self.id, self.base_ip, self.size, self.location, self.status)