from __future__ import annotations
import configparser
from ....util import (
from . import (
class OpenNebulaCloudEnvironment(CloudEnvironment):
    """Updates integration test environment after delegation. Will setup the config file as parameter."""

    def get_environment_config(self) -> CloudEnvironmentConfig:
        """Return environment configuration for use in the test environment after delegation."""
        parser = configparser.ConfigParser()
        parser.read(self.config_path)
        ansible_vars = dict(resource_prefix=self.resource_prefix)
        ansible_vars.update(dict(parser.items('default')))
        display.sensitive.add(ansible_vars.get('opennebula_password'))
        return CloudEnvironmentConfig(ansible_vars=ansible_vars)