import copy
import os.path
import typing as ty
from urllib import parse
import warnings
from keystoneauth1 import discover
import keystoneauth1.exceptions.catalog
from keystoneauth1.loading import adapter as ks_load_adap
from keystoneauth1 import session as ks_session
import os_service_types
import requestsexceptions
from openstack import _log
from openstack.config import _util
from openstack.config import defaults as config_defaults
from openstack import exceptions
from openstack import proxy
from openstack import version as openstack_version
from openstack import warnings as os_warnings
def _get_hardcoded_endpoint(self, service_type, constructor):
    endpoint = self._get_endpoint_from_catalog(service_type, constructor)
    if not endpoint.rstrip().rsplit('/')[-1] == 'v2.0':
        if not endpoint.endswith('/'):
            endpoint += '/'
        endpoint = parse.urljoin(endpoint, 'v2.0')
    return endpoint