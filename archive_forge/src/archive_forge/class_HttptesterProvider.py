from __future__ import annotations
import os
from ....util import (
from ....config import (
from ....containers import (
from . import (
class HttptesterProvider(CloudProvider):
    """HTTP Tester provider plugin. Sets up resources before delegation."""

    def __init__(self, args: IntegrationConfig) -> None:
        super().__init__(args)
        self.image = os.environ.get('ANSIBLE_HTTP_TEST_CONTAINER', 'quay.io/ansible/http-test-container:2.1.0')
        self.uses_docker = True

    def setup(self) -> None:
        """Setup resources before delegation."""
        super().setup()
        ports = [80, 88, 443, 444, 749]
        aliases = ['ansible.http.tests', 'sni1.ansible.http.tests', 'fail.ansible.http.tests', 'self-signed.ansible.http.tests']
        descriptor = run_support_container(self.args, self.platform, self.image, 'http-test-container', ports, aliases=aliases, env={KRB5_PASSWORD_ENV: generate_password()})
        if not descriptor:
            return
        krb5_password = descriptor.details.container.env_dict()[KRB5_PASSWORD_ENV]
        display.sensitive.add(krb5_password)
        self._set_cloud_config(KRB5_PASSWORD_ENV, krb5_password)