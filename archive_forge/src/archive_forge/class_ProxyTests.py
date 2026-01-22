from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
class ProxyTests(unittest.TestCase):

    def setUp(self):
        self.proxy = sip.Proxy('127.0.0.1')
        self.proxy.locator = DummyLocator()
        self.sent = []
        self.proxy.sendMessage = lambda dest, msg: self.sent.append((dest, msg))

    def testRequestForward(self):
        r = sip.Request('INVITE', 'sip:foo')
        r.addHeader('via', sip.Via('1.2.3.4').toString())
        r.addHeader('via', sip.Via('1.2.3.5').toString())
        r.addHeader('foo', 'bar')
        r.addHeader('to', '<sip:joe@server.com>')
        r.addHeader('contact', '<sip:joe@1.2.3.5>')
        self.proxy.datagramReceived(r.toString(), ('1.2.3.4', 5060))
        self.assertEqual(len(self.sent), 1)
        dest, m = self.sent[0]
        self.assertEqual(dest.port, 5060)
        self.assertEqual(dest.host, 'server.com')
        self.assertEqual(m.uri.toString(), 'sip:foo')
        self.assertEqual(m.method, 'INVITE')
        self.assertEqual(m.headers['via'], ['SIP/2.0/UDP 127.0.0.1:5060', 'SIP/2.0/UDP 1.2.3.4:5060', 'SIP/2.0/UDP 1.2.3.5:5060'])

    def testReceivedRequestForward(self):
        r = sip.Request('INVITE', 'sip:foo')
        r.addHeader('via', sip.Via('1.2.3.4').toString())
        r.addHeader('foo', 'bar')
        r.addHeader('to', '<sip:joe@server.com>')
        r.addHeader('contact', '<sip:joe@1.2.3.4>')
        self.proxy.datagramReceived(r.toString(), ('1.1.1.1', 5060))
        dest, m = self.sent[0]
        self.assertEqual(m.headers['via'], ['SIP/2.0/UDP 127.0.0.1:5060', 'SIP/2.0/UDP 1.2.3.4:5060;received=1.1.1.1'])

    def testResponseWrongVia(self):
        r = sip.Response(200)
        r.addHeader('via', sip.Via('foo.com').toString())
        self.proxy.datagramReceived(r.toString(), ('1.1.1.1', 5060))
        self.assertEqual(len(self.sent), 0)

    def testResponseForward(self):
        r = sip.Response(200)
        r.addHeader('via', sip.Via('127.0.0.1').toString())
        r.addHeader('via', sip.Via('client.com', port=1234).toString())
        self.proxy.datagramReceived(r.toString(), ('1.1.1.1', 5060))
        self.assertEqual(len(self.sent), 1)
        dest, m = self.sent[0]
        self.assertEqual((dest.host, dest.port), ('client.com', 1234))
        self.assertEqual(m.code, 200)
        self.assertEqual(m.headers['via'], ['SIP/2.0/UDP client.com:1234'])

    def testReceivedResponseForward(self):
        r = sip.Response(200)
        r.addHeader('via', sip.Via('127.0.0.1').toString())
        r.addHeader('via', sip.Via('10.0.0.1', received='client.com').toString())
        self.proxy.datagramReceived(r.toString(), ('1.1.1.1', 5060))
        self.assertEqual(len(self.sent), 1)
        dest, m = self.sent[0]
        self.assertEqual((dest.host, dest.port), ('client.com', 5060))

    def testResponseToUs(self):
        r = sip.Response(200)
        r.addHeader('via', sip.Via('127.0.0.1').toString())
        l = []
        self.proxy.gotResponse = lambda *a: l.append(a)
        self.proxy.datagramReceived(r.toString(), ('1.1.1.1', 5060))
        self.assertEqual(len(l), 1)
        m, addr = l[0]
        self.assertEqual(len(m.headers.get('via', [])), 0)
        self.assertEqual(m.code, 200)

    def testLoop(self):
        r = sip.Request('INVITE', 'sip:foo')
        r.addHeader('via', sip.Via('1.2.3.4').toString())
        r.addHeader('via', sip.Via('127.0.0.1').toString())
        self.proxy.datagramReceived(r.toString(), ('client.com', 5060))
        self.assertEqual(self.sent, [])

    def testCantForwardRequest(self):
        r = sip.Request('INVITE', 'sip:foo')
        r.addHeader('via', sip.Via('1.2.3.4').toString())
        r.addHeader('to', '<sip:joe@server.com>')
        self.proxy.locator = FailingLocator()
        self.proxy.datagramReceived(r.toString(), ('1.2.3.4', 5060))
        self.assertEqual(len(self.sent), 1)
        dest, m = self.sent[0]
        self.assertEqual((dest.host, dest.port), ('1.2.3.4', 5060))
        self.assertEqual(m.code, 404)
        self.assertEqual(m.headers['via'], ['SIP/2.0/UDP 1.2.3.4:5060'])