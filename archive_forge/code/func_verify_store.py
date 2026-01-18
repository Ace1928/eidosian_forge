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
def verify_store():
    store_id = CONF.glance_store.default_backend
    if not store_id:
        msg = _("'default_backend' config option is not set.")
        raise RuntimeError(msg)
    try:
        get_store_from_store_identifier(store_id)
    except exceptions.UnknownScheme:
        msg = _('Store for identifier %s not found') % store_id
        raise RuntimeError(msg)