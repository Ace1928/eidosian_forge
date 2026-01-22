import itertools
import eventlet
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import netutils
import tenacity
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
from heat.engine.resources.openstack.neutron import port as neutron_port
def _str_network(self, networks):
    for net in networks or []:
        str_net = net.get(self.ALLOCATE_NETWORK)
        if str_net:
            return str_net