import os
import time
from cinderclient.v3 import client as cinderclient
import fixtures
from glanceclient import client as glanceclient
from keystoneauth1.exceptions import discovery as discovery_exc
from keystoneauth1 import identity
from keystoneauth1 import session as ksession
from keystoneclient import client as keystoneclient
from keystoneclient import discover as keystone_discover
from neutronclient.v2_0 import client as neutronclient
import openstack.config
import openstack.config.exceptions
from oslo_utils import uuidutils
import tempest.lib.cli.base
import testtools
import novaclient
import novaclient.api_versions
from novaclient import base
import novaclient.client
from novaclient.v2 import networks
import novaclient.v2.shell
def _get_novaclient(self, session):
    nc = novaclient.client.Client('2', session=session)
    if self.COMPUTE_API_VERSION:
        if 'min_api_version' not in CACHE:
            v = nc.versions.get_current()
            if not hasattr(v, 'version') or not v.version:
                CACHE['min_api_version'] = novaclient.api_versions.APIVersion('2.0')
                CACHE['max_api_version'] = novaclient.api_versions.APIVersion('2.0')
            else:
                CACHE['min_api_version'] = novaclient.api_versions.APIVersion(v.min_version)
                CACHE['max_api_version'] = novaclient.api_versions.APIVersion(v.version)
        if self.COMPUTE_API_VERSION == '2.latest':
            requested_version = min(novaclient.API_MAX_VERSION, CACHE['max_api_version'])
        else:
            requested_version = novaclient.api_versions.APIVersion(self.COMPUTE_API_VERSION)
        if not requested_version.matches(CACHE['min_api_version'], CACHE['max_api_version']):
            msg = '%s is not supported by Nova-API. Supported version' % self.COMPUTE_API_VERSION
            if CACHE['min_api_version'] == CACHE['max_api_version']:
                msg += ': %s' % CACHE['min_api_version'].get_string()
            else:
                msg += 's: %s - %s' % (CACHE['min_api_version'].get_string(), CACHE['max_api_version'].get_string())
            self.skipTest(msg)
        nc.api_version = requested_version
    return nc