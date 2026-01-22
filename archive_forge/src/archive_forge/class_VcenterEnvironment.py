from __future__ import annotations
import configparser
from ....util import (
from ....config import (
from . import (
class VcenterEnvironment(CloudEnvironment):
    """VMware vcenter/esx environment plugin. Updates integration test environment after delegation."""

    def get_environment_config(self) -> CloudEnvironmentConfig:
        """Return environment configuration for use in the test environment after delegation."""
        parser = configparser.ConfigParser()
        parser.read(self.config_path)
        ansible_vars = dict(resource_prefix=self.resource_prefix)
        ansible_vars.update(dict(parser.items('DEFAULT', raw=True)))
        for key, value in ansible_vars.items():
            if key.endswith('_password'):
                display.sensitive.add(value)
        return CloudEnvironmentConfig(ansible_vars=ansible_vars, module_defaults={'group/vmware': {'hostname': ansible_vars['vcenter_hostname'], 'username': ansible_vars['vcenter_username'], 'password': ansible_vars['vcenter_password'], 'port': ansible_vars.get('vcenter_port', '443'), 'validate_certs': ansible_vars.get('vmware_validate_certs', 'no')}})