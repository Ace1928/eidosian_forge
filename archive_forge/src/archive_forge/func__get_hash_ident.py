import itertools
from oslo_log import log
import passlib.hash
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _get_hash_ident(hashers):
    for hasher in hashers:
        if hasattr(hasher, 'prefix'):
            ident = (getattr(hasher, 'prefix'),)
        elif hasattr(hasher, 'ident_values'):
            ident = getattr(hasher, 'ident_values')
        else:
            ident = (getattr(hasher, 'ident'),)
        yield (hasher, ident)