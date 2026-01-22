from monascaclient import exc as monasca_exc
from monascaclient.v2_0 import client as monasca_client
from heat.common import exception as heat_exc
from heat.engine.clients import client_plugin
from heat.engine import constraints
class MonascaClientPlugin(client_plugin.ClientPlugin):
    exceptions_module = [monasca_exc]
    service_types = [MONITORING] = ['monitoring']
    VERSION = '2_0'

    def _create(self):
        interface = self._get_client_option(CLIENT_NAME, 'endpoint_type')
        endpoint = self.url_for(service_type=self.MONITORING, endpoint_type=interface)
        return monasca_client.Client(session=self.context.keystone_session, service_type='monitoring', endpoint=endpoint)

    def is_not_found(self, ex):
        return isinstance(ex, monasca_exc.NotFound)

    def is_un_processable(self, ex):
        return isinstance(ex, monasca_exc.UnprocessableEntity)

    def get_notification(self, notification):
        try:
            return self.client().notifications.get(notification_id=notification)['id']
        except monasca_exc.NotFound:
            raise heat_exc.EntityNotFound(entity='Monasca Notification', name=notification)