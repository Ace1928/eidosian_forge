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
def use_conn_pool(func):
    """Use this only for connection pool specific ldap API.

    This adds connection object to decorated API as next argument after self.

    """

    def wrapper(self, *args, **kwargs):
        with self._get_pool_connection() as conn:
            self._apply_options(conn)
            return func(self, conn, *args, **kwargs)
    return wrapper