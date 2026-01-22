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
def _get_version_request(self, service_type, version):
    """Translate OCC version args to those needed by ksa adapter.

        If no version is requested explicitly and we have a configured version,
        set the version parameter and let ksa deal with expanding that to
        min=ver.0, max=ver.latest.

        If version is set, pass it through.

        If version is not set and we don't have a configured version, default
        to latest.

        If version is set, contains a '.', and default_microversion is not
        set, also pass it as a default microversion.
        """
    version_request = _util.VersionRequest()
    if version == 'latest':
        version_request.max_api_version = 'latest'
        return version_request
    if not version:
        version = self.get_api_version(service_type)
    if not version and service_type not in ('load-balancer',):
        version_request.max_api_version = 'latest'
    else:
        version_request.version = version
    default_microversion = self.get_default_microversion(service_type)
    implied_microversion = _get_implied_microversion(version)
    if implied_microversion and default_microversion and (implied_microversion != default_microversion):
        raise exceptions.ConfigException('default_microversion of {default_microversion} was given for {service_type}, but api_version looks like a microversion as well. Please set api_version to just the desired major version, or omit default_microversion'.format(default_microversion=default_microversion, service_type=service_type))
    if implied_microversion:
        default_microversion = implied_microversion
        version_request.version = version[0]
    version_request.default_microversion = default_microversion
    return version_request