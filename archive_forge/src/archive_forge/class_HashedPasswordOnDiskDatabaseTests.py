from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
class HashedPasswordOnDiskDatabaseTests(unittest.TestCase):
    users = [(b'user1', b'pass1'), (b'user2', b'pass2'), (b'user3', b'pass3')]

    def setUp(self):
        dbfile = self.mktemp()
        self.db = checkers.FilePasswordDB(dbfile, hash=self.hash)
        with open(dbfile, 'wb') as f:
            for u, p in self.users:
                f.write(u + b':' + self.hash(u, p, u[:2]) + b'\n')
        r = TestRealm()
        self.port = portal.Portal(r)
        self.port.registerChecker(self.db)

    def hash(self, u: bytes, p: bytes, s: bytes) -> bytes:
        hashed_password = crypt(p.decode('ascii'), s.decode('ascii'))
        return hashed_password.encode('ascii')

    def testGoodCredentials(self):
        goodCreds = [credentials.UsernamePassword(u, p) for u, p in self.users]
        d = defer.gatherResults([self.db.requestAvatarId(c) for c in goodCreds])
        d.addCallback(self.assertEqual, [u for u, p in self.users])
        return d

    def testGoodCredentials_login(self):
        goodCreds = [credentials.UsernamePassword(u, p) for u, p in self.users]
        d = defer.gatherResults([self.port.login(c, None, ITestable) for c in goodCreds])
        d.addCallback(lambda x: [a.original.name for i, a, l in x])
        d.addCallback(self.assertEqual, [u for u, p in self.users])
        return d

    def testBadCredentials(self):
        badCreds = [credentials.UsernamePassword(u, b'wrong password') for u, p in self.users]
        d = defer.DeferredList([self.port.login(c, None, ITestable) for c in badCreds], consumeErrors=True)
        d.addCallback(self._assertFailures, error.UnauthorizedLogin)
        return d

    def testHashedCredentials(self):
        UsernameHashedPassword = self.getDeprecatedModuleAttribute('twisted.cred.credentials', 'UsernameHashedPassword', _uhpVersion)
        hashedCreds = [UsernameHashedPassword(u, self.hash(None, p, u[:2])) for u, p in self.users]
        d = defer.DeferredList([self.port.login(c, None, ITestable) for c in hashedCreds], consumeErrors=True)
        d.addCallback(self._assertFailures, error.UnhandledCredentials)
        return d

    def _assertFailures(self, failures, *expectedFailures):
        for flag, failure in failures:
            self.assertEqual(flag, defer.FAILURE)
            failure.trap(*expectedFailures)
        return None
    if crypt is None:
        skip = 'crypt module not available'