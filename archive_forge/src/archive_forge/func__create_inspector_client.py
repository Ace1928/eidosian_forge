import logging
import sys
from cliff import app
from cliff import commandmanager
import openstack
from openstack import config as os_config
from osc_lib import utils
import pbr.version
from ironicclient.common import http
from ironicclient.common.i18n import _
from ironicclient import exc
from ironicclient.v1 import client
def _create_inspector_client(self):
    assert ironic_inspector_client is not None, 'BUG: _create_inspector_client called without inspector client'
    endpoint_override = self.cloud_region.get_endpoint(_INSPECTOR_TYPE)
    try:
        return ironic_inspector_client.ClientV1(inspector_url=endpoint_override, session=self.cloud_region.get_session(), region_name=self.cloud_region.get_region_name(_INSPECTOR_TYPE))
    except ironic_inspector_client.EndpointNotFound as e:
        raise exc.EndpointNotFound(_HELP % {'err': e, 'cmd': sys.argv[0], 'project': 'ironic-inspector'})