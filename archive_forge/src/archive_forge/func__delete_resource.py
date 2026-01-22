import copy
import itertools
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import output
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.heat import resource_group
from heat.engine.resources import signal_responder
from heat.engine import rsrc_defn
from heat.engine import software_config_io as swc_io
from heat.engine import support
from heat.rpc import api as rpc_api
def _delete_resource(self):
    derived_config_id = None
    if self.resource_id is not None:
        with self.rpc_client().ignore_error_by_name('NotFound'):
            derived_config_id = self._get_derived_config_id()
            self.rpc_client().delete_software_deployment(self.context, self.resource_id)
    if derived_config_id:
        self._delete_derived_config(derived_config_id)
    self._delete_signals()
    self._delete_user()