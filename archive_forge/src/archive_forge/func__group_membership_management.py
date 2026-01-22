from contextlib import contextmanager
from typing import cast
import logging
from . import api
from . import TensorPipeAgent
@contextmanager
def _group_membership_management(store, name, is_join):
    token_key = 'RpcGroupManagementToken'
    join_or_leave = 'join' if is_join else 'leave'
    my_token = f'Token_for_{name}_{join_or_leave}'
    while True:
        returned = store.compare_set(token_key, '', my_token).decode()
        if returned == my_token:
            yield
            store.set(token_key, '')
            store.set(my_token, 'Done')
            break
        else:
            try:
                store.wait([returned])
            except RuntimeError:
                logger.error('Group membership token %s timed out waiting for %s to be released.', my_token, returned)
                raise