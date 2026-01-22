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
def _not_found(self, object_id):
    if self.NotFound is None:
        return exception.NotFound(target=object_id)
    else:
        return self.NotFound(**{self.notfound_arg: object_id})