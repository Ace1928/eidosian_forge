from __future__ import (absolute_import, division, print_function)
import datetime
from contextlib import contextmanager
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.plugins.cache import BaseCacheModule
from ansible.utils.display import Display
from ansible.module_utils._text import to_native
def _ttl_index_exists(self, collection):
    """
        Returns true if an index named ttl exists
        on the given collection.
        """
    exists = False
    try:
        indexes = collection.list_indexes()
        for index in indexes:
            if index['name'] == 'ttl':
                exists = True
                break
    except pymongo.errors.OperationFailure as excep:
        raise AnsibleError('Error checking MongoDB index: %s' % to_native(excep))
    return exists