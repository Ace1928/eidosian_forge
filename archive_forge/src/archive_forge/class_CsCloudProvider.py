from __future__ import annotations
import json
import configparser
import os
import urllib.parse
import typing as t
from ....util import (
from ....config import (
from ....docker_util import (
from ....containers import (
from . import (
class CsCloudProvider(CloudProvider):
    """CloudStack cloud provider plugin. Sets up cloud resources before delegation."""

    def __init__(self, args: IntegrationConfig) -> None:
        super().__init__(args)
        self.image = os.environ.get('ANSIBLE_CLOUDSTACK_CONTAINER', 'quay.io/ansible/cloudstack-test-container:1.6.1')
        self.host = ''
        self.port = 0
        self.uses_docker = True
        self.uses_config = True

    def setup(self) -> None:
        """Setup the cloud resource before delegation and register a cleanup callback."""
        super().setup()
        if self._use_static_config():
            self._setup_static()
        else:
            self._setup_dynamic()

    def _setup_static(self) -> None:
        """Configure CloudStack tests for use with static configuration."""
        parser = configparser.ConfigParser()
        parser.read(self.config_static_path)
        endpoint = parser.get('cloudstack', 'endpoint')
        parts = urllib.parse.urlparse(endpoint)
        self.host = parts.hostname
        if not self.host:
            raise ApplicationError('Could not determine host from endpoint: %s' % endpoint)
        if parts.port:
            self.port = parts.port
        elif parts.scheme == 'http':
            self.port = 80
        elif parts.scheme == 'https':
            self.port = 443
        else:
            raise ApplicationError('Could not determine port from endpoint: %s' % endpoint)
        display.info('Read cs host "%s" and port %d from config: %s' % (self.host, self.port, self.config_static_path), verbosity=1)

    def _setup_dynamic(self) -> None:
        """Create a CloudStack simulator using docker."""
        config = self._read_config_template()
        self.port = 8888
        ports = [self.port]
        descriptor = run_support_container(self.args, self.platform, self.image, 'cloudstack-sim', ports)
        if not descriptor:
            return
        docker_exec(self.args, descriptor.name, ['find', '/var/lib/mysql', '-type', 'f', '-exec', 'touch', '{}', ';'], capture=True)
        if self.args.explain:
            values = dict(HOST=self.host, PORT=str(self.port))
        else:
            credentials = self._get_credentials(descriptor.name)
            values = dict(HOST=descriptor.name, PORT=str(self.port), KEY=credentials['apikey'], SECRET=credentials['secretkey'])
            display.sensitive.add(values['SECRET'])
        config = self._populate_config_template(config, values)
        self._write_config(config)

    def _get_credentials(self, container_name: str) -> dict[str, t.Any]:
        """Wait for the CloudStack simulator to return credentials."""

        def check(value) -> bool:
            """Return True if the given configuration is valid JSON, otherwise return False."""
            try:
                json.loads(value)
            except Exception:
                return False
            return True
        stdout = wait_for_file(self.args, container_name, '/var/www/html/admin.json', sleep=10, tries=30, check=check)
        return json.loads(stdout)