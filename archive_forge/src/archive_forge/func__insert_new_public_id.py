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
def _insert_new_public_id(self, local_entity, ref, driver):
    public_id = None
    if driver.generates_uuids():
        public_id = ref['id']
    ref['id'] = PROVIDERS.id_mapping_api.create_id_mapping(local_entity, public_id)
    LOG.debug('Created new mapping to public ID: %s', ref['id'])