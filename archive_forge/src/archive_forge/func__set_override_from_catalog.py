import typing as ty
import warnings
import os_service_types
from openstack import _log
from openstack import exceptions
from openstack import proxy as proxy_mod
from openstack import warnings as os_warnings
def _set_override_from_catalog(self, config):
    override = config._get_endpoint_from_catalog(self.service_type, proxy_mod.Proxy)
    config.set_service_value('endpoint_override', self.service_type, override)