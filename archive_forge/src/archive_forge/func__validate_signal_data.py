import copy
from oslo_serialization import jsonutils
import yaml
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import signal_responder
from heat.engine import support
from heat.engine import translation
def _validate_signal_data(self, inputs, params):
    if inputs is not None:
        if not isinstance(inputs, dict):
            message = _('Input in signal data must be a map, find a %s') % type(inputs)
            raise exception.StackValidationFailed(error=_('Signal data error'), message=message)
        for key in inputs:
            if self.properties.get(self.INPUT) is None or key not in self.properties.get(self.INPUT):
                message = _('Unknown input %s') % key
                raise exception.StackValidationFailed(error=_('Signal data error'), message=message)
    if params is not None and (not isinstance(params, dict)):
        message = _('Params must be a map, find a %s') % type(params)
        raise exception.StackValidationFailed(error=_('Signal data error'), message=message)