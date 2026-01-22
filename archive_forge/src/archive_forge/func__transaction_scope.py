import contextlib
import functools
import inspect
import operator
import threading
import warnings
import debtcollector.moves
import debtcollector.removals
import debtcollector.renames
from oslo_config import cfg
from oslo_utils import excutils
from oslo_db import exception
from oslo_db import options
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import orm
from oslo_db import warning
@contextlib.contextmanager
def _transaction_scope(self, context):
    new_transaction = self._independent
    transaction_contexts_by_thread = _transaction_contexts_by_thread(context)
    current = restore = getattr(transaction_contexts_by_thread, 'current', None)
    use_factory = self._factory
    global_factory = None
    if self._replace_global_factory:
        use_factory = global_factory = self._replace_global_factory
    elif current is not None and current.global_factory:
        global_factory = current.global_factory
        if self._root._is_global_manager:
            use_factory = global_factory
    if current is not None and (new_transaction or current.factory is not use_factory):
        current = None
    if current is None:
        current = transaction_contexts_by_thread.current = _TransactionContext(use_factory, global_factory=global_factory)
    try:
        if self._mode is not None:
            with current._produce_block(mode=self._mode, connection=self._connection, savepoint=self._savepoint, allow_async=self._allow_async, context=context) as resource:
                yield resource
        else:
            yield
    finally:
        if restore is None:
            del transaction_contexts_by_thread.current
        elif current is not restore:
            transaction_contexts_by_thread.current = restore