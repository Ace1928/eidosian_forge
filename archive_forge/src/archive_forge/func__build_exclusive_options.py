from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import scheduler_hints as sh
from heat.engine.resources import volume_base as vb
from heat.engine import support
from heat.engine import translation
def _build_exclusive_options(self):
    exclusive_options = []
    allow_no_size_options = []
    if self.properties.get(self.SNAPSHOT_ID):
        exclusive_options.append(self.SNAPSHOT_ID)
        allow_no_size_options.append(self.SNAPSHOT_ID)
    if self.properties.get(self.SOURCE_VOLID):
        exclusive_options.append(self.SOURCE_VOLID)
        allow_no_size_options.append(self.SOURCE_VOLID)
    if self.properties.get(self.IMAGE):
        exclusive_options.append(self.IMAGE)
    if self.properties.get(self.IMAGE_REF):
        exclusive_options.append(self.IMAGE_REF)
    return (exclusive_options, allow_no_size_options)