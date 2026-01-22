from urllib import parse
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import password_gen
from heat.engine.clients.os import swift
from heat.engine.resources import stack_user
def _get_zaqar_signal_queue_id(self):
    """Return a zaqar queue_id for signaling this resource.

        This uses the created user for the credentials.
        """
    queue_id = self.data().get('zaqar_signal_queue_id')
    if queue_id:
        return queue_id
    if self.id is None:
        return
    if self._get_user_id() is None:
        if self.password is None:
            self.password = password_gen.generate_openstack_password()
        self._create_user()
    queue_id = self.physical_resource_name()
    zaqar_plugin = self.client_plugin('zaqar')
    zaqar = zaqar_plugin.create_for_tenant(self.stack.stack_user_project_id, self._user_token())
    queue = zaqar.queue(queue_id)
    signed_url_data = queue.signed_url(['messages'], methods=['GET', 'DELETE'])
    self.data_set('zaqar_queue_signed_url_data', jsonutils.dumps(signed_url_data))
    self.data_set('zaqar_signal_queue_id', queue_id)
    return queue_id