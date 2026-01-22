import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def _format_dns_nameservers(subnet):
    try:
        return '\n'.join([jsonutils.dumps(server) for server in subnet['dns_nameservers']])
    except (TypeError, KeyError):
        return ''