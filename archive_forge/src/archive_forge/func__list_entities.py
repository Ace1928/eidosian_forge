import uuid
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def _list_entities(self, entity_type):
    """Find the list_<entity_type> method.

        Searches through the [identity_api, resource_api, assignment_api]
        managers for a method called list_<entity_type> and returns the first
        one.

        """
    f = getattr(PROVIDERS.identity_api, 'list_%ss' % entity_type, None)
    if f is None:
        f = getattr(PROVIDERS.resource_api, 'list_%ss' % entity_type, None)
    if f is None:
        f = getattr(PROVIDERS.assignment_api, 'list_%ss' % entity_type)
    return f