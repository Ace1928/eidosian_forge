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
def _create_factory_copy(self):
    factory = _TransactionFactory()
    factory._url_cfg.update(self._url_cfg)
    factory._engine_cfg.update(self._engine_cfg)
    factory._maker_cfg.update(self._maker_cfg)
    factory._transaction_ctx_cfg.update(self._transaction_ctx_cfg)
    factory._facade_cfg.update(self._facade_cfg)
    return factory