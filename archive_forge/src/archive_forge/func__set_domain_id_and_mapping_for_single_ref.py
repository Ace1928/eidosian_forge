import copy
import functools
import itertools
import operator
import os
import threading
import uuid
from oslo_config import cfg
from oslo_log import log
from pycadf import reason
from keystone import assignment  # TODO(lbragstad): Decouple this dependency
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.mapping_backends import mapping
from keystone import notifications
from oslo_utils import timeutils
def _set_domain_id_and_mapping_for_single_ref(self, ref, domain_id, driver, entity_type, conf):
    LOG.debug('Local ID: %s', ref['id'])
    ref = ref.copy()
    if not driver.is_domain_aware():
        if not domain_id:
            domain_id = CONF.identity.default_domain_id
        ref['domain_id'] = domain_id
    if self._is_mapping_needed(driver):
        local_entity = {'domain_id': ref['domain_id'], 'local_id': ref['id'], 'entity_type': entity_type}
        public_id = PROVIDERS.id_mapping_api.get_public_id(local_entity)
        if public_id:
            ref['id'] = public_id
            LOG.debug('Found existing mapping to public ID: %s', ref['id'])
        else:
            self._insert_new_public_id(local_entity, ref, driver)
    return ref