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
def _apply_options(self, conn):
    if conn.get_lifetime() > 30:
        return
    for option, invalue in self.conn_options.items():
        conn.set_option(option, invalue)