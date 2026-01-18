import random
import re
import shelve
import ldap
from oslo_log import log
import keystone.conf
from keystone import exception
from keystone.identity.backends.ldap import common
def unbind_ext_s(self):
    """Added to extend FakeLdap as connector class."""
    pass