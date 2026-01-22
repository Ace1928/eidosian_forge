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
def _update_user_with_federated_objects(self, user, driver, entity_id):
    if not user.get('federated'):
        if 'federated' in user:
            del user['federated']
        user = driver.update_user(entity_id, user)
        fed_objects = self.shadow_users_api.get_federated_objects(user['id'])
        if fed_objects:
            user['federated'] = fed_objects
        return user
    else:
        user_ref = user.copy()
        self._validate_federated_objects(user_ref['federated'])
        self.shadow_users_api.delete_federated_object(entity_id)
        del user['federated']
        user = driver.update_user(entity_id, user)
        self._create_federated_objects(user, user_ref['federated'])
        user['federated'] = user_ref['federated']
        return user