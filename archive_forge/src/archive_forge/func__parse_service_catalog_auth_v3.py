import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def _parse_service_catalog_auth_v3(self, service_catalog):
    entries = []
    for item in service_catalog:
        service_type = item['type']
        service_name = item.get('name', None)
        entry_endpoints = []
        for endpoint in item['endpoints']:
            region = endpoint.get('region', None)
            url = endpoint['url']
            endpoint_type = endpoint['interface']
            if endpoint_type == 'internal':
                endpoint_type = OpenStackIdentityEndpointType.INTERNAL
            elif endpoint_type == 'public':
                endpoint_type = OpenStackIdentityEndpointType.EXTERNAL
            elif endpoint_type == 'admin':
                endpoint_type = OpenStackIdentityEndpointType.ADMIN
            entry_endpoint = OpenStackServiceCatalogEntryEndpoint(region=region, url=url, endpoint_type=endpoint_type)
            entry_endpoints.append(entry_endpoint)
        entry = OpenStackServiceCatalogEntry(service_type=service_type, service_name=service_name, endpoints=entry_endpoints)
        entries.append(entry)
    return entries