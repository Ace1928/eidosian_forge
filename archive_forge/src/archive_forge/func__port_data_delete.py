import copy
from oslo_config import cfg
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import scheduler_hints as sh
def _port_data_delete(self):
    port_id = self.data().get('port_id')
    if port_id:
        with self.client_plugin('neutron').ignore_not_found:
            self.neutron().delete_port(port_id)
        self.data_delete('port_id')