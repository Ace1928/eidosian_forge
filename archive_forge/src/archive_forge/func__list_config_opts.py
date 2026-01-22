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
def _list_config_opts():
    driver_opts = _list_driver_opts()
    sample_opts = [(_STORE_CFG_GROUP, _STORE_OPTS)]
    for store_entry in driver_opts:
        if store_entry == 'no_conf':
            continue
        sample_opts.append((store_entry, driver_opts[store_entry]))
    return sample_opts