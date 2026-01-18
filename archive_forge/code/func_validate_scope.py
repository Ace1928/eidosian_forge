import copy
import logging
from openstack.config import loader as config  # noqa
from openstack import connection
from oslo_utils import strutils
from osc_lib.api import auth
from osc_lib import exceptions
def validate_scope(self):
    if self._auth_ref.project_id is not None:
        return
    if self._auth_ref.domain_id is not None:
        return
    auth.check_valid_authorization_options(self._cli_options, self.auth_plugin_name)