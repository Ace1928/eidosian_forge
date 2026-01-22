from datetime import datetime
from datetime import timedelta
from unittest import mock
from oslo_config import cfg
from heat.common import template_format
from heat.engine import environment
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import snapshot as snapshot_objects
from heat.objects import stack as stack_object
from heat.objects import sync_point as sync_point_object
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def _mock_convg_db_update_requires(self):
    """Updates requires column of resources.

        Required for testing the generation of convergence dependency graph
        on an update.
        """
    requires = dict()
    for rsrc_id, is_update in self.stack.convergence_dependencies:
        if is_update:
            reqs = self.stack.convergence_dependencies.requires((rsrc_id, is_update))
            requires[rsrc_id] = list({id for id, is_update in reqs})
    rsrcs_db = resource_objects.Resource.get_all_active_by_stack(self.stack.context, self.stack.id)
    for rsrc_id, rsrc in rsrcs_db.items():
        if rsrc.id in requires:
            rsrcs_db[rsrc_id].requires = requires[rsrc.id]
    return rsrcs_db