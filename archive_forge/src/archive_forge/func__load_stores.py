import hashlib
import logging
from oslo_config import cfg
from oslo_utils import encodeutils
from stevedore import driver
from stevedore import extension
from glance_store import capabilities
from glance_store import exceptions
from glance_store.i18n import _
from glance_store import location
def _load_stores(conf):
    for store_entry in set(conf.glance_store.stores):
        try:
            store_instance = _load_store(conf, store_entry)
            if not store_instance:
                continue
            yield (store_entry, store_instance)
        except exceptions.BadStoreConfiguration:
            continue