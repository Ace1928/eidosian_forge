from oslo_log import log as logging
from zaqarclient.queues.v2 import client as zaqarclient
from zaqarclient.transport import errors as zaqar_errors
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine import constraints
class ZaqarClientPlugin(client_plugin.ClientPlugin):
    exceptions_module = zaqar_errors
    service_types = [MESSAGING] = ['messaging']
    DEFAULT_TTL = 3600

    def _create(self):
        return zaqarclient.Client(version=2, session=self.context.keystone_session)

    def create_for_tenant(self, tenant_id, token):
        con = self.context
        if token is None:
            LOG.error('Zaqar connection failed, no auth_token!')
            return None
        opts = {'os_auth_token': token, 'os_auth_url': con.auth_url, 'os_project_id': tenant_id, 'os_service_type': self.MESSAGING}
        auth_opts = {'backend': 'keystone', 'options': opts}
        conf = {'auth_opts': auth_opts}
        endpoint = self.url_for(service_type=self.MESSAGING)
        return zaqarclient.Client(url=endpoint, conf=conf, version=2)

    def create_from_signed_url(self, project_id, paths, expires, methods, signature):
        opts = {'paths': paths, 'expires': expires, 'methods': methods, 'signature': signature, 'os_project_id': project_id}
        auth_opts = {'backend': 'signed-url', 'options': opts}
        conf = {'auth_opts': auth_opts}
        endpoint = self.url_for(service_type=self.MESSAGING)
        return zaqarclient.Client(url=endpoint, conf=conf, version=2)

    def is_not_found(self, ex):
        return isinstance(ex, zaqar_errors.ResourceNotFound)

    def get_queue(self, queue_name):
        if not isinstance(queue_name, str):
            raise TypeError(_('Queue name must be a string'))
        if not 0 < len(queue_name) <= 64:
            raise ValueError(_('Queue name length must be 1-64'))
        return queue_name