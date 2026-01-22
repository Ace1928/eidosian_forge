import copy
import logging
import threading
from debtcollector import removals
from oslo_config import cfg
from oslo_db import api
@removals.removed_class('TpoolDbapiWrapper', message=_removed_msg, version='10.0.0')
class TpoolDbapiWrapper(object):
    """DB API wrapper class.

    This wraps the oslo DB API with an option to be able to use eventlet's
    thread pooling. Since the CONF variable may not be loaded at the time
    this class is instantiated, we must look at it on the first DB API call.
    """

    def __init__(self, conf, backend_mapping):
        self._db_api = None
        self._backend_mapping = backend_mapping
        self._conf = conf
        self._conf.register_opts(tpool_opts, 'database')
        self._lock = threading.Lock()

    @property
    def _api(self):
        if not self._db_api:
            with self._lock:
                if not self._db_api:
                    db_api = api.DBAPI.from_config(conf=self._conf, backend_mapping=self._backend_mapping)
                    if self._conf.database.use_tpool:
                        try:
                            from eventlet import tpool
                        except ImportError:
                            LOG.exception("'eventlet' is required for TpoolDbapiWrapper.")
                            raise
                        self._db_api = tpool.Proxy(db_api)
                    else:
                        self._db_api = db_api
        return self._db_api

    def __getattr__(self, key):
        return getattr(self._api, key)