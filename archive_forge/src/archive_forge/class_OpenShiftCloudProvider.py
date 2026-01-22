from __future__ import annotations
import re
from ....io import (
from ....util import (
from ....config import (
from ....containers import (
from . import (
class OpenShiftCloudProvider(CloudProvider):
    """OpenShift cloud provider plugin. Sets up cloud resources before delegation."""

    def __init__(self, args: IntegrationConfig) -> None:
        super().__init__(args, config_extension='.kubeconfig')
        self.image = 'quay.io/ansible/openshift-origin:v3.9.0'
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
        """Configure OpenShift tests for use with static configuration."""
        config = read_text_file(self.config_static_path)
        match = re.search('^ *server: (?P<server>.*)$', config, flags=re.MULTILINE)
        if not match:
            display.warning('Could not find OpenShift endpoint in kubeconfig.')

    def _setup_dynamic(self) -> None:
        """Create a OpenShift container using docker."""
        port = 8443
        ports = [port]
        cmd = ['start', 'master', '--listen', 'https://0.0.0.0:%d' % port]
        descriptor = run_support_container(self.args, self.platform, self.image, 'openshift-origin', ports, cmd=cmd)
        if not descriptor:
            return
        if self.args.explain:
            config = '# Unknown'
        else:
            config = self._get_config(descriptor.name, 'https://%s:%s/' % (descriptor.name, port))
        self._write_config(config)

    def _get_config(self, container_name: str, server: str) -> str:
        """Get OpenShift config from container."""
        stdout = wait_for_file(self.args, container_name, '/var/lib/origin/openshift.local.config/master/admin.kubeconfig', sleep=10, tries=30)
        config = stdout
        config = re.sub('^( *)certificate-authority-data: .*$', '\\1insecure-skip-tls-verify: true', config, flags=re.MULTILINE)
        config = re.sub('^( *)server: .*$', '\\1server: %s' % server, config, flags=re.MULTILINE)
        return config