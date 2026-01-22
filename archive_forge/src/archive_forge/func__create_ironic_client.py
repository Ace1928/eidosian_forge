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
def _create_ironic_client(self):
    api_version = self.options.os_baremetal_api_version
    allow_api_version_downgrade = False
    if not api_version:
        api_version = self.cloud_region.get_default_microversion(_TYPE)
        if not api_version:
            api_version = http.LATEST_VERSION
            allow_api_version_downgrade = True
    LOG.debug('Using bare metal API version %s, downgrade %s', api_version, 'allowed' if allow_api_version_downgrade else 'disallowed')
    endpoint_override = self.cloud_region.get_endpoint(_TYPE)
    try:
        return client.Client(os_ironic_api_version=api_version, allow_api_version_downgrade=allow_api_version_downgrade, session=self.cloud_region.get_session(), region_name=self.cloud_region.get_region_name(_TYPE), endpoint_override=endpoint_override, max_retries=self.options.max_retries, retry_interval=self.options.retry_interval)
    except exc.EndpointNotFound as e:
        raise exc.EndpointNotFound(_HELP % {'err': e, 'cmd': sys.argv[0], 'project': 'ironic'})