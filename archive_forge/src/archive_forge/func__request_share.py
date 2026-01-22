from oslo_log import log as logging
from oslo_utils import encodeutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def _request_share(self):
    return self.client().shares.get(self.resource_id)