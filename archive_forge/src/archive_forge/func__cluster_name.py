import re
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
def _cluster_name(self):
    name = self.properties[self.NAME]
    if name:
        return name
    return self.reduce_physical_resource_name(re.sub('[^a-zA-Z0-9-]', '', self.physical_resource_name()), SAHARA_CLUSTER_NAME_MAX_LENGTH)