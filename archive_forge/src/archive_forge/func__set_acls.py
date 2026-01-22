from collections import abc
import copy
import functools
from cryptography import exceptions as crypto_exception
from cursive import exception as cursive_exception
from cursive import signature_utils
import glance_store as store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from glance.common import exception
from glance.common import format_inspector
from glance.common import store_utils
from glance.common import utils
import glance.domain.proxy
from glance.i18n import _, _LE, _LI, _LW
def _set_acls(self):
    public = self.image.visibility in ['public', 'community']
    if self.image.locations and (not public):
        member_ids = [m.member_id for m in self.repo.list()]
        for location in self.image.locations:
            if CONF.enabled_backends:
                self.store_api.set_acls_for_multi_store(location['url'], location['metadata'].get('store'), public=public, read_tenants=member_ids, context=self.context)
            else:
                self.store_api.set_acls(location['url'], public=public, read_tenants=member_ids, context=self.context)