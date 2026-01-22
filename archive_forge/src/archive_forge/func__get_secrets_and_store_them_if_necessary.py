import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
from barbicanclient.v1 import secrets as secret_manager
def _get_secrets_and_store_them_if_necessary(self):
    LOG.debug('Storing secrets: {0}'.format(base.censored_copy(self.secrets, ['payload'])))
    secret_refs = []
    for name, secret in self.secrets.items():
        if secret and (not secret.secret_ref):
            secret.store()
        secret_refs.append({'name': name, 'secret_ref': secret.secret_ref})
    return secret_refs