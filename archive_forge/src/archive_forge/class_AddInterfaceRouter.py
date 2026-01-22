import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import availability_zone
class AddInterfaceRouter(RouterInterfaceCommand):
    """Add an internal network interface to a router."""

    def call_api(self, neutron_client, router_id, body):
        return neutron_client.add_interface_router(router_id, body)

    def success_message(self, router_id, portinfo):
        return _('Added interface %(port)s to router %(router)s.') % {'router': router_id, 'port': portinfo['port_id']}