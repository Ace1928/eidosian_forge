import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import password_gen
from heat.engine.clients import progress
from heat.engine.resources import stack_user
def _create_transport_credentials(self, props):
    if self.transport_poll_server_cfn(props):
        self._create_user()
        self._create_keypair()
    elif self.transport_poll_server_heat(props) or self.transport_zaqar_message(props):
        if self.password is None:
            self.password = password_gen.generate_openstack_password()
        self._create_user()
    self._register_access_key()