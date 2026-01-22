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
def _persist_revocation_event_for_user(self, user_id):
    """Emit a notification to invoke a revocation event callback.

        Fire off an internal notification that will be consumed by the
        revocation API to store a revocation record for a specific user.

        :param user_id: user identifier
        :type user_id: string
        """
    notifications.Audit.internal(notifications.PERSIST_REVOCATION_EVENT_FOR_USER, user_id)