import collections
import email
from email.mime import multipart
from email.mime import text
import os
import pkgutil
import string
from urllib import parse as urlparse
from neutronclient.common import exceptions as q_exceptions
from novaclient import api_versions
from novaclient import client as nc
from novaclient import exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import netutils
import tenacity
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_exception
from heat.engine.clients import client_plugin
from heat.engine.clients import microversion_mixin
from heat.engine.clients import os as os_client
from heat.engine import constraints
def _check_active(self, server, res_name='Server'):
    """Check server status.

        Accepts both server IDs and server objects.
        Returns True if server is ACTIVE,
        raises errors when server has an ERROR or unknown to Heat status,
        returns False otherwise.

        :param res_name: name of the resource to use in the exception message

        """
    if isinstance(server, str):
        server = self.fetch_server(server)
        if server is None:
            return False
        else:
            status = self.get_status(server)
    else:
        status = self.get_status(server)
        if status != 'ACTIVE':
            self.refresh_server(server)
            status = self.get_status(server)
    if status in self.deferred_server_statuses:
        return False
    elif status == 'ACTIVE':
        return True
    elif status == 'ERROR':
        fault = getattr(server, 'fault', {})
        raise exception.ResourceInError(resource_status=status, status_reason=_('Message: %(message)s, Code: %(code)s') % {'message': fault.get('message', _('Unknown')), 'code': fault.get('code', _('Unknown'))})
    else:
        raise exception.ResourceUnknownStatus(resource_status=server.status, result=_('%s is not active') % res_name)