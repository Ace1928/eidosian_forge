from __future__ import annotations
import configparser
from ....util import (
from ....config import (
from ....target import (
from ....core_ci import (
from . import (
class HcloudCloudProvider(CloudProvider):
    """Hetzner Cloud provider plugin. Sets up cloud resources before delegation."""

    def __init__(self, args: IntegrationConfig) -> None:
        super().__init__(args)
        self.uses_config = True

    def filter(self, targets: tuple[IntegrationTarget, ...], exclude: list[str]) -> None:
        """Filter out the cloud tests when the necessary config and resources are not available."""
        aci = self._create_ansible_core_ci()
        if aci.available:
            return
        super().filter(targets, exclude)

    def setup(self) -> None:
        """Setup the cloud resource before delegation and register a cleanup callback."""
        super().setup()
        if not self._use_static_config():
            self._setup_dynamic()

    def _setup_dynamic(self) -> None:
        """Request Hetzner credentials through the Ansible Core CI service."""
        display.info('Provisioning %s cloud environment.' % self.platform, verbosity=1)
        config = self._read_config_template()
        aci = self._create_ansible_core_ci()
        response = aci.start()
        if not self.args.explain:
            token = response['hetzner']['token']
            display.sensitive.add(token)
            display.info('Hetzner Cloud Token: %s' % token, verbosity=1)
            values = dict(TOKEN=token)
            display.sensitive.add(values['TOKEN'])
            config = self._populate_config_template(config, values)
        self._write_config(config)

    def _create_ansible_core_ci(self) -> AnsibleCoreCI:
        """Return a Heztner instance of AnsibleCoreCI."""
        return AnsibleCoreCI(self.args, CloudResource(platform='hetzner'))