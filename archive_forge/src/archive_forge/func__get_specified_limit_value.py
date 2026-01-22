from oslo_log import log
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
from keystone.limit.models import base
def _get_specified_limit_value(self, resource_name, service_id, region_id, project_id=None, domain_id=None):
    """Get the specified limit value.

        Try to give the resource limit first. If the specified limit is a
        domain limit and the resource limit value is None, get the related
        registered limit value instead.

        """
    hints = driver_hints.Hints()
    if project_id:
        hints.add_filter('project_id', project_id)
    else:
        hints.add_filter('domain_id', domain_id)
    hints.add_filter('service_id', service_id)
    hints.add_filter('resource_name', resource_name)
    hints.add_filter('region_id', region_id)
    limits = PROVIDERS.unified_limit_api.list_limits(hints)
    limit_value = limits[0]['resource_limit'] if limits else None
    if not limits and domain_id:
        hints = driver_hints.Hints()
        hints.add_filter('service_id', service_id)
        hints.add_filter('resource_name', resource_name)
        hints.add_filter('region_id', region_id)
        limits = PROVIDERS.unified_limit_api.list_registered_limits(hints)
        limit_value = limits[0]['default_limit'] if limits else None
    return limit_value