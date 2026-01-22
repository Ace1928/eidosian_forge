from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import properties
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import support
def _job_binary_name(self):
    return self.properties[self.NAME] or self.physical_resource_name()