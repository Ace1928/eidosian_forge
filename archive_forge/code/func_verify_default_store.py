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
def verify_default_store():
    scheme = CONF.glance_store.default_store
    try:
        get_store_from_scheme(scheme)
    except exceptions.UnknownScheme:
        msg = _('Store for scheme %s not found') % scheme
        raise RuntimeError(msg)