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
def _get_domain_driver_and_entity_id(self, public_id):
    """Look up details using the public ID.

        :param public_id: the ID provided in the call

        :returns: domain_id, which can be None to indicate that the driver
                  in question supports multiple domains
                  driver selected based on this domain
                  entity_id which will is understood by the driver.

        Use the mapping table to look up the domain, driver and local entity
        that is represented by the provided public ID.  Handle the situations
        where we do not use the mapping (e.g. single driver that understands
        UUIDs etc.)

        """
    conf = CONF.identity
    if conf.domain_specific_drivers_enabled:
        local_id_ref = PROVIDERS.id_mapping_api.get_id_mapping(public_id)
        if local_id_ref:
            return (local_id_ref['domain_id'], self._select_identity_driver(local_id_ref['domain_id']), local_id_ref['local_id'])
    driver = self.driver
    if driver.generates_uuids():
        if driver.is_domain_aware:
            return (None, driver, public_id)
        else:
            return (conf.default_domain_id, driver, public_id)
    if not CONF.identity_mapping.backward_compatible_ids:
        local_id_ref = PROVIDERS.id_mapping_api.get_id_mapping(public_id)
        if local_id_ref:
            return (local_id_ref['domain_id'], driver, local_id_ref['local_id'])
        else:
            raise exception.PublicIDNotFound(id=public_id)
    return (conf.default_domain_id, driver, public_id)