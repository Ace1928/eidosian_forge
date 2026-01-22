import functools
from heat.common import exception
from heat.common.i18n import _
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import template as cfn_template
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine.hot import parameters
from heat.engine import rsrc_defn
from heat.engine import template_common
def _translate_resources(self, resources):
    """Get the resources of the template translated into CFN format."""
    return self._translate_section(self.RESOURCES, self.RES_TYPE, resources, self._RESOURCE_HOT_TO_CFN_ATTRS)