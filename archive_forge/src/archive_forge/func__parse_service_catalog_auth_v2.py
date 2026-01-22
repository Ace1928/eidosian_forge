import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def _parse_service_catalog_auth_v2(self, service_catalog):
    entries = []
    for service in service_catalog:
        service_type = service['type']
        service_name = service.get('name', None)
        entry_endpoints = []
        for endpoint in service.get('endpoints', []):
            region = endpoint.get('region', None)
            public_url = endpoint.get('publicURL', None)
            private_url = endpoint.get('internalURL', None)
            if public_url:
                entry_endpoint = OpenStackServiceCatalogEntryEndpoint(region=region, url=public_url, endpoint_type=OpenStackIdentityEndpointType.EXTERNAL)
                entry_endpoints.append(entry_endpoint)
            if private_url:
                entry_endpoint = OpenStackServiceCatalogEntryEndpoint(region=region, url=private_url, endpoint_type=OpenStackIdentityEndpointType.INTERNAL)
                entry_endpoints.append(entry_endpoint)
        entry = OpenStackServiceCatalogEntry(service_type=service_type, endpoints=entry_endpoints, service_name=service_name)
        entries.append(entry)
    return entries