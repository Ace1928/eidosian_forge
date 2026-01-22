import abc
import copy
from keystoneauth1 import discover
from keystoneauth1 import exceptions
def _denormalize_endpoints(self, endpoints):
    """Return original endpoint description dicts.

        Takes a list of EndpointData objects and returns the original
        dict that was returned from the catalog.

        :param list endpoints: List of `keystoneauth1.discover.EndpointData`

        :returns: List of endpoint description dicts in original catalog format
        """
    raw_endpoints = super(ServiceCatalogV2, self)._denormalize_endpoints(endpoints)
    seen = {}
    endpoints = []
    for endpoint in raw_endpoints:
        if str(endpoint) in seen:
            continue
        seen[str(endpoint)] = True
        endpoints.append(endpoint)
    return endpoints