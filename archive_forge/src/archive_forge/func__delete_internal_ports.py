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
def _delete_internal_ports(self):
    for port_data in self._data_get_ports():
        self._delete_internal_port(port_data['id'])
    self.data_delete('internal_ports')