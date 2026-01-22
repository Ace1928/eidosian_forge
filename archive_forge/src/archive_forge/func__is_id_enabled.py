import abc
import codecs
import os.path
import random
import re
import sys
import uuid
import weakref
import ldap.controls
import ldap.filter
import ldappool
from oslo_log import log
from oslo_utils import reflection
from keystone.common import driver_hints
from keystone import exception
from keystone.i18n import _
def _is_id_enabled(self, object_id, conn):
    member_attr_val = self._id_to_member_attribute_value(object_id)
    return self._is_member_enabled(member_attr_val, conn)