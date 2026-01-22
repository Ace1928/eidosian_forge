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
class MappingManager(manager.Manager):
    """Default pivot point for the ID Mapping backend."""
    driver_namespace = 'keystone.identity.id_mapping'
    _provides_api = 'id_mapping_api'

    def __init__(self):
        super(MappingManager, self).__init__(CONF.identity_mapping.driver)

    @MEMOIZE_ID_MAPPING
    def _get_public_id(self, domain_id, local_id, entity_type):
        return self.driver.get_public_id({'domain_id': domain_id, 'local_id': local_id, 'entity_type': entity_type})

    def get_public_id(self, local_entity):
        return self._get_public_id(local_entity['domain_id'], local_entity['local_id'], local_entity['entity_type'])

    @MEMOIZE_ID_MAPPING
    def get_id_mapping(self, public_id):
        return self.driver.get_id_mapping(public_id)

    def create_id_mapping(self, local_entity, public_id=None):
        public_id = self.driver.create_id_mapping(local_entity, public_id)
        if MEMOIZE_ID_MAPPING.should_cache(public_id):
            self._get_public_id.set(public_id, self, local_entity['domain_id'], local_entity['local_id'], local_entity['entity_type'])
            self.get_id_mapping.set(local_entity, self, public_id)
        return public_id

    def delete_id_mapping(self, public_id):
        local_entity = self.get_id_mapping.get(self, public_id)
        self.driver.delete_id_mapping(public_id)
        if local_entity:
            self._get_public_id.invalidate(self, local_entity['domain_id'], local_entity['local_id'], local_entity['entity_type'])
        self.get_id_mapping.invalidate(self, public_id)

    def purge_mappings(self, purge_filter):
        self.driver.purge_mappings(purge_filter)
        ID_MAPPING_REGION.invalidate()