from novaclient import base
from novaclient import exceptions
from novaclient.i18n import _
class NeutronManager(base.Manager):
    """A manager for name -> id lookups for neutron networks.

    This uses neutron directly from service catalog. Do not use it
    for anything else besides that. You have been warned.
    """
    resource_class = Network

    def find_network(self, name):
        """Find a network by name (user provided input)."""
        with self.alternate_service_type('network', allowed_types=('network',)):
            matches = self._list('/v2.0/networks?name=%s' % name, 'networks')
            num_matches = len(matches)
            if num_matches == 0:
                msg = 'No %s matching %s.' % (self.resource_class.__name__, name)
                raise exceptions.NotFound(404, msg)
            elif num_matches > 1:
                msg = _("Multiple %(class)s matches found for '%(name)s', use an ID to be more specific.") % {'class': self.resource_class.__name__.lower(), 'name': name}
                raise exceptions.NoUniqueMatch(msg)
            else:
                matches[0].append_request_ids(matches.request_ids)
                return matches[0]