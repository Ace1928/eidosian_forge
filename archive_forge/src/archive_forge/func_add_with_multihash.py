import copy
import hashlib
import logging
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import units
from stevedore import driver
from stevedore import extension
from glance_store import capabilities
from glance_store import exceptions
from glance_store.i18n import _
from glance_store import location
def add_with_multihash(conf, image_id, data, size, backend, hashing_algo, scheme=None, context=None, verifier=None):
    if not backend:
        backend = conf.glance_store.default_backend
    store = get_store_from_store_identifier(backend)
    return store_add_to_backend_with_multihash(image_id, data, size, hashing_algo, store, context, verifier)