import uuid
from oslo_log import log
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.server import backends
def _bootstrap_catalog(self):
    if self.public_url or self.admin_url or self.internal_url:
        hints = driver_hints.Hints()
        hints.add_filter('type', 'identity')
        services = PROVIDERS.catalog_api.list_services(hints)
        if services:
            service = services[0]
            hints = driver_hints.Hints()
            hints.add_filter('service_id', service['id'])
            if self.region_id:
                hints.add_filter('region_id', self.region_id)
            endpoints = PROVIDERS.catalog_api.list_endpoints(hints)
        else:
            service_id = uuid.uuid4().hex
            service = {'id': service_id, 'name': self.service_name, 'type': 'identity', 'enabled': True}
            PROVIDERS.catalog_api.create_service(service_id, service)
            endpoints = []
        self.service_id = service['id']
        available_interfaces = {e['interface']: e for e in endpoints}
        expected_endpoints = {'public': self.public_url, 'internal': self.internal_url, 'admin': self.admin_url}
        for interface, url in expected_endpoints.items():
            if not url:
                continue
            try:
                endpoint_ref = available_interfaces[interface]
            except KeyError:
                endpoint_ref = {'id': uuid.uuid4().hex, 'interface': interface, 'url': url, 'service_id': self.service_id, 'enabled': True}
                if self.region_id:
                    endpoint_ref['region_id'] = self.region_id
                PROVIDERS.catalog_api.create_endpoint(endpoint_id=endpoint_ref['id'], endpoint_ref=endpoint_ref)
                LOG.info('Created %(interface)s endpoint %(url)s', {'interface': interface, 'url': url})
            else:
                endpoint_ref['url'] = url
                PROVIDERS.catalog_api.update_endpoint(endpoint_id=endpoint_ref['id'], endpoint_ref=endpoint_ref)
                LOG.info('%s endpoint updated', interface)
            self.endpoints[interface] = endpoint_ref['id']