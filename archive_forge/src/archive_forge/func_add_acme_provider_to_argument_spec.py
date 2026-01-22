from __future__ import absolute_import, division, print_function
import os
import tempfile
import traceback
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate import (
def add_acme_provider_to_argument_spec(argument_spec):
    argument_spec.argument_spec['provider']['choices'].append('acme')
    argument_spec.argument_spec.update(dict(acme_accountkey_path=dict(type='path'), acme_challenge_path=dict(type='path'), acme_chain=dict(type='bool', default=False), acme_directory=dict(type='str', default='https://acme-v02.api.letsencrypt.org/directory')))