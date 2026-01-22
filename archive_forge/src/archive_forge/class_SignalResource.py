import collections
from oslo_log import log as logging
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import signal_responder
from heat.engine.resources import stack_resource
from heat.engine.resources import stack_user
from heat.engine import support
class SignalResource(signal_responder.SignalResponder):
    SIGNAL_TRANSPORTS = CFN_SIGNAL, TEMP_URL_SIGNAL, HEAT_SIGNAL, NO_SIGNAL, ZAQAR_SIGNAL = ('CFN_SIGNAL', 'TEMP_URL_SIGNAL', 'HEAT_SIGNAL', 'NO_SIGNAL', 'ZAQAR_SIGNAL')
    properties_schema = {'signal_transport': properties.Schema(properties.Schema.STRING, default='CFN_SIGNAL')}
    attributes_schema = {'AlarmUrl': attributes.Schema('Get a signed webhook'), 'signal': attributes.Schema('Get a signal')}

    def handle_create(self):
        self.password = 'password'
        super(SignalResource, self).handle_create()
        self.resource_id_set(self._get_user_id())

    def handle_signal(self, details=None):
        LOG.warning('Signaled resource (Type "%(type)s") %(details)s', {'type': self.type(), 'details': details})

    def _resolve_attribute(self, name):
        if self.resource_id is not None:
            if name == 'AlarmUrl':
                return self._get_signal().get('alarm_url')
            elif name == 'signal':
                return self._get_signal()