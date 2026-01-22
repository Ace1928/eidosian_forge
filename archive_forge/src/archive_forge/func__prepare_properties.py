import re
from oslo_log import log as logging
from oslo_utils import encodeutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import support
from heat.engine import translation
def _prepare_properties(self, props):
    """Prepares the property values."""
    if self.NAME in props:
        props['name'] = self._cluster_template_name(props[self.NAME])
    if self.MANAGEMENT_NETWORK in props:
        props['net_id'] = props.pop(self.MANAGEMENT_NETWORK)
    return props