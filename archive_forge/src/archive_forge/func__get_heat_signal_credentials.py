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
def _get_heat_signal_credentials(self):
    """Return OpenStack credentials that can be used to send a signal.

        These credentials are for the user associated with this resource in
        the heat stack user domain.
        """
    if self._get_user_id() is None:
        if self.password is None:
            self.password = password_gen.generate_openstack_password()
        self._create_user()
    return {'auth_url': self.keystone().server_keystone_endpoint_url(fallback_endpoint=self.keystone().v3_endpoint), 'username': self.physical_resource_name(), 'user_id': self._get_user_id(), 'password': self.password, 'project_id': self.stack.stack_user_project_id, 'domain_id': self.keystone().stack_domain_id, 'region_name': self._get_region_name()}