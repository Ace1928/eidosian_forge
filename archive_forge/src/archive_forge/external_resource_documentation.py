from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
Check execution status.

        Returns False if in IDLE, RUNNING or PAUSED
        returns True if in SUCCESS
        raises ResourceFailure if in ERROR, CANCELLED
        raises ResourceUnknownState otherwise.
        