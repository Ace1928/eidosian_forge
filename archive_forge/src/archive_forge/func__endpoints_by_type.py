import abc
import copy
from keystoneauth1 import discover
from keystoneauth1 import exceptions
def _endpoints_by_type(self, requested, endpoints):
    """Get the approrpriate endpoints from the list of given endpoints.

        Per the service type alias rules:

        If a user requests a service by its proper name and that matches, win.

        If a user requests a service by its proper name and only a single alias
        matches, win.

        If a user requests a service by its proper name and more than one alias
        matches, choose the first alias from the list given.

        Do the "first alias" match after the other filters, as they might limit
        the number of choices for us otherwise.

        :param str requested:
            The service_type as requested by the user.
        :param dict sc:
            A dictionary keyed by found service_type. Values are opaque to
            this method.

        :returns:
            Dict of service_type/endpoints filtered for the appropriate
            service_type based on alias matching rules.
        """
    if not requested or not discover._SERVICE_TYPES.is_known(requested):
        return endpoints
    if len(endpoints) < 2:
        return endpoints
    if endpoints.get(requested):
        return {requested: endpoints[requested]}
    for alias in discover._SERVICE_TYPES.get_all_types(requested):
        if endpoints.get(alias):
            return {alias: endpoints[alias]}
    raise ValueError('Programming error choosing an endpoint.')