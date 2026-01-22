from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as nc
from oslo_config import cfg
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
def _resolve_resource_path(self, resource):
    """Returns ext resource path."""
    if resource == 'port_pair':
        path = '/sfc/port_pairs'
    elif resource == 'port_pair_group':
        path = '/sfc/port_pair_groups'
    elif resource == 'flow_classifier':
        path = '/sfc/flow_classifiers'
    elif resource == 'port_chain':
        path = '/sfc/port_chains'
    elif resource == 'tap_service':
        path = '/taas/tap_services'
    elif resource == 'tap_flow':
        path = '/taas/tap_flows'
    return path