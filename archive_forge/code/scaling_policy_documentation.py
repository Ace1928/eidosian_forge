from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import signal_responder
from heat.engine import support
from heat.scaling import scalingutil as sc_util
Updates self.properties, if Properties has changed.

        If Properties has changed, update self.properties, so we get the new
        values during any subsequent adjustment.
        