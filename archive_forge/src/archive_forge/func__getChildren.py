import random
import re
import shelve
import ldap
from oslo_log import log
import keystone.conf
from keystone import exception
from keystone.identity.backends.ldap import common
def _getChildren(self, dn):
    return [k for k, v in self.db.items() if re.match('%s.*,%s' % (re.escape(self.__prefix), re.escape(dn)), k)]