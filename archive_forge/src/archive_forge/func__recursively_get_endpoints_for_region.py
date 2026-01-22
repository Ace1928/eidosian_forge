from oslo_log import log
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _recursively_get_endpoints_for_region(region_id, service_id, endpoint_list, region_list, endpoints_found, regions_examined):
    """Recursively search down a region tree for endpoints.

                :param region_id: the point in the tree to examine
                :param service_id: the service we are interested in
                :param endpoint_list: list of all endpoints
                :param region_list: list of all regions
                :param endpoints_found: list of matching endpoints found so
                                        far - which will be updated if more are
                                        found in this iteration
                :param regions_examined: list of regions we have already looked
                                         at - used to spot illegal circular
                                         references in the tree to avoid never
                                         completing search
                :returns: list of endpoints that match

                """
    if region_id in regions_examined:
        msg = 'Circular reference or a repeated entry found in region tree - %(region_id)s.'
        LOG.error(msg, {'region_id': ref.region_id})
        return
    regions_examined.append(region_id)
    endpoints_found += [ep for ep in endpoint_list if ep['service_id'] == service_id and ep['region_id'] == region_id]
    for region in region_list:
        if region['parent_region_id'] == region_id:
            _recursively_get_endpoints_for_region(region['id'], service_id, endpoints, regions, endpoints_found, regions_examined)