from __future__ import annotations
import os
import tempfile
from ....config import (
from ....docker_util import (
from ....containers import (
from ....encoding import (
from ....util import (
from . import (
class GalaxyProvider(CloudProvider):
    """
    Galaxy plugin. Sets up pulp (ansible-galaxy) servers for tests.
    The pulp source itself resides at: https://github.com/pulp/pulp-oci-images
    """

    def __init__(self, args: IntegrationConfig) -> None:
        super().__init__(args)
        self.image = os.environ.get('ANSIBLE_PULP_CONTAINER', 'quay.io/pulp/galaxy:4.7.1')
        self.uses_docker = True

    def setup(self) -> None:
        """Setup cloud resource before delegation and reg cleanup callback."""
        super().setup()
        with tempfile.NamedTemporaryFile(mode='w+') as env_fd:
            settings = '\n'.join((f'{key}={value}' for key, value in SETTINGS.items()))
            env_fd.write(settings)
            env_fd.flush()
            display.info(f'>>> galaxy_ng Configuration\n{settings}', verbosity=3)
            descriptor = run_support_container(self.args, self.platform, self.image, GALAXY_HOST_NAME, [80], aliases=[GALAXY_HOST_NAME], start=True, options=['--env-file', env_fd.name])
        if not descriptor:
            return
        injected_files = [('/etc/galaxy-importer/galaxy-importer.cfg', GALAXY_IMPORTER, 'galaxy-importer')]
        for path, content, friendly_name in injected_files:
            with tempfile.NamedTemporaryFile() as temp_fd:
                temp_fd.write(content)
                temp_fd.flush()
                display.info(f'>>> {friendly_name} Configuration\n{to_text(content)}', verbosity=3)
                docker_exec(self.args, descriptor.container_id, ['mkdir', '-p', os.path.dirname(path)], True)
                docker_cp_to(self.args, descriptor.container_id, temp_fd.name, path)
                docker_exec(self.args, descriptor.container_id, ['chown', 'pulp:pulp', path], True)
        self._set_cloud_config('PULP_HOST', GALAXY_HOST_NAME)
        self._set_cloud_config('PULP_USER', 'admin')
        self._set_cloud_config('PULP_PASSWORD', 'password')