from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import progress
from heat.engine import resource
from heat.engine import rsrc_defn
def _create_backup(self):
    backup = self.client().backups.create(self.resource_id)
    return backup.id