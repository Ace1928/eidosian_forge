import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackServiceCatalogEntry:

    def __init__(self, service_type, endpoints=None, service_name=None):
        """
        :param service_type: Service type.
        :type service_type: ``str``

        :param endpoints: Endpoints belonging to this entry.
        :type endpoints: ``list``

        :param service_name: Optional service name.
        :type service_name: ``str``
        """
        self.service_type = service_type
        self.endpoints = endpoints or []
        self.service_name = service_name
        self.endpoints = sorted(self.endpoints, key=lambda x: x.url or '')

    def __eq__(self, other):
        return self.service_type == other.service_type and self.endpoints == other.endpoints and (other.service_name == self.service_name)

    def __ne__(self, other):
        return not self.__eq__(other=other)

    def __repr__(self):
        return '<OpenStackServiceCatalogEntry service_type=%s, service_name=%s, endpoints=%s' % (self.service_type, self.service_name, repr(self.endpoints))