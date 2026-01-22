from __future__ import absolute_import, division, print_function
import os
from random import randrange
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate import (
def add_selfsigned_provider_to_argument_spec(argument_spec):
    argument_spec.argument_spec['provider']['choices'].append('selfsigned')
    argument_spec.argument_spec.update(dict(selfsigned_version=dict(type='int', default=3), selfsigned_digest=dict(type='str', default='sha256'), selfsigned_not_before=dict(type='str', default='+0s', aliases=['selfsigned_notBefore']), selfsigned_not_after=dict(type='str', default='+3650d', aliases=['selfsigned_notAfter']), selfsigned_create_subject_key_identifier=dict(type='str', default='create_if_not_provided', choices=['create_if_not_provided', 'always_create', 'never_create'])))