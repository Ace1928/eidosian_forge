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
class ShadowUsersManager(manager.Manager):
    """Default pivot point for the Shadow Users backend."""
    driver_namespace = 'keystone.identity.shadow_users'
    _provides_api = 'shadow_users_api'

    def __init__(self):
        shadow_driver = CONF.shadow_users.driver
        super(ShadowUsersManager, self).__init__(shadow_driver)