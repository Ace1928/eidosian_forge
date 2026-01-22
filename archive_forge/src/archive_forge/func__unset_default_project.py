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
def _unset_default_project(self, service, resource_type, operation, payload):
    """Callback, clears user default_project_id after project deletion.

        Notifications are used to unset a user's default project because
        there is no foreign key to the project. Projects can be in a non-SQL
        backend, making FKs impossible.

        """
    project_id = payload['resource_info']
    drivers = itertools.chain(self.domain_configs.values(), [{'driver': self.driver}])
    for d in drivers:
        try:
            d['driver'].unset_default_project_id(project_id)
        except exception.Forbidden:
            pass