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
def _id_to_dn_string(self, object_id):
    return u'%s=%s,%s' % (self.id_attr, ldap.dn.escape_dn_chars(str(object_id)), self.tree_dn)