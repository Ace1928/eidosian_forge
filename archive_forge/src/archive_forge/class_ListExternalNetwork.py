import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import availability_zone
from neutronclient.neutron.v2_0 import dns
from neutronclient.neutron.v2_0.qos import policy as qos_policy
class ListExternalNetwork(ListNetwork):
    """List external networks that belong to a given tenant."""
    pagination_support = True
    sorting_support = True

    def retrieve_list(self, parsed_args):
        external = '--router:external=True'
        if external not in self.values_specs:
            self.values_specs.append('--router:external=True')
        return super(ListExternalNetwork, self).retrieve_list(parsed_args)