from __future__ import absolute_import, division, print_function
import base64
import json
import os
import traceback
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api import auth
from ansible_collections.community.docker.plugins.module_utils._api.auth import decode_auth
from ansible_collections.community.docker.plugins.module_utils._api.credentials.errors import CredentialsNotFound
from ansible_collections.community.docker.plugins.module_utils._api.credentials.store import Store
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
def update_credentials(self):
    """
        If the authorization is not stored attempt to store authorization values via
        the appropriate credential helper or to the config file.

        :return: None
        """
    store = self.get_credential_store_instance(self.registry_url, self.config_path)
    try:
        current = store.get(self.registry_url)
    except CredentialsNotFound:
        current = dict(Username='', Secret='')
    if current['Username'] != self.username or current['Secret'] != self.password or self.reauthorize:
        if not self.check_mode:
            store.store(self.registry_url, self.username, self.password)
        self.log('Writing credentials to configured helper %s for %s' % (store.program, self.registry_url))
        self.results['actions'].append('Wrote credentials to configured helper %s for %s' % (store.program, self.registry_url))
        self.results['changed'] = True