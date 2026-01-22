from unittest import skipIf
from twisted.internet import defer, reactor
from twisted.trial.unittest import TestCase
from twisted.web import error, server
@skipIf(not SOAPpy, 'SOAPpy not installed')
class SOAPTests(TestCase):

    def setUp(self):
        self.publisher = Test()
        self.p = reactor.listenTCP(0, server.Site(self.publisher), interface='127.0.0.1')
        self.port = self.p.getHost().port

    def tearDown(self):
        return self.p.stopListening()

    def proxy(self):
        return soap.Proxy('http://127.0.0.1:%d/' % self.port)

    def testResults(self):
        inputOutput = [('add', (2, 3), 5), ('defer', ('a',), 'a'), ('dict', ({'a': 1}, 'a'), 1), ('triple', ('a', 1), ['a', 1, None])]
        dl = []
        for meth, args, outp in inputOutput:
            d = self.proxy().callRemote(meth, *args)
            d.addCallback(self.assertEqual, outp)
            dl.append(d)
        d = self.proxy().callRemote('complex')
        d.addCallback(lambda result: result._asdict())
        d.addCallback(self.assertEqual, {'a': ['b', 'c', 12, []], 'D': 'foo'})
        dl.append(d)
        return defer.DeferredList(dl, fireOnOneErrback=True)

    def testMethodNotFound(self):
        """
        Check that a non existing method return error 500.
        """
        d = self.proxy().callRemote('doesntexist')
        self.assertFailure(d, error.Error)

        def cb(err):
            self.assertEqual(int(err.status), 500)
        d.addCallback(cb)
        return d

    def testLookupFunction(self):
        """
        Test lookupFunction method on publisher, to see available remote
        methods.
        """
        self.assertTrue(self.publisher.lookupFunction('add'))
        self.assertTrue(self.publisher.lookupFunction('fail'))
        self.assertFalse(self.publisher.lookupFunction('foobar'))