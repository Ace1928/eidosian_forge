from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def _data_source_name(self):
    return self.properties[self.NAME] or self.physical_resource_name()