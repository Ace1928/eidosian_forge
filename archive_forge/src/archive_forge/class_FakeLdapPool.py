import random
import re
import shelve
import ldap
from oslo_log import log
import keystone.conf
from keystone import exception
from keystone.identity.backends.ldap import common
class FakeLdapPool(FakeLdap):
    """Emulate the python-ldap API with pooled connections.

    This class is used as connector class in PooledLDAPHandler.

    """

    def __init__(self, uri, retry_max=None, retry_delay=None, conn=None):
        super(FakeLdapPool, self).__init__(conn=conn)
        self.url = uri
        self._uri = uri
        self.connected = None
        self.conn = self
        self._connection_time = 5

    def get_lifetime(self):
        return self._connection_time

    def simple_bind_s(self, who=None, cred=None, serverctrls=None, clientctrls=None):
        if self.url.startswith('fakepool://memory'):
            if self.url not in FakeShelves:
                FakeShelves[self.url] = FakeShelve()
            self.db = FakeShelves[self.url]
        else:
            self.db = shelve.open(self.url[11:])
        if not who:
            who = 'cn=Admin'
        if not cred:
            cred = 'password'
        super(FakeLdapPool, self).simple_bind_s(who=who, cred=cred, serverctrls=serverctrls, clientctrls=clientctrls)

    def unbind_ext_s(self):
        """Added to extend FakeLdap as connector class."""
        pass