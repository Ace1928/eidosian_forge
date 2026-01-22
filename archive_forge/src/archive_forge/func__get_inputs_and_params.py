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
def _get_inputs_and_params(self, data):
    inputs = None
    params = None
    if self.properties.get(self.USE_REQUEST_BODY_AS_INPUT):
        inputs = data
    elif data is not None:
        inputs = data.get(self.SIGNAL_DATA_INPUT)
        params = data.get(self.SIGNAL_DATA_PARAMS)
    return (inputs, params)