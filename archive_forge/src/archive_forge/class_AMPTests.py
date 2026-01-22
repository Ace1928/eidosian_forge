import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
class AMPTests(TestCase):

    def test_interfaceDeclarations(self):
        """
        The classes in the amp module ought to implement the interfaces that
        are declared for their benefit.
        """
        for interface, implementation in [(amp.IBoxSender, amp.BinaryBoxProtocol), (amp.IBoxReceiver, amp.BoxDispatcher), (amp.IResponderLocator, amp.CommandLocator), (amp.IResponderLocator, amp.SimpleStringLocator), (amp.IBoxSender, amp.AMP), (amp.IBoxReceiver, amp.AMP), (amp.IResponderLocator, amp.AMP)]:
            self.assertTrue(interface.implementedBy(implementation), f'{implementation} does not implements({interface})')

    def test_helloWorld(self):
        """
        Verify that a simple command can be sent and its response received with
        the simple low-level string-based API.
        """
        c, s, p = connectedServerAndClient()
        L = []
        HELLO = b'world'
        c.sendHello(HELLO).addCallback(L.append)
        p.flush()
        self.assertEqual(L[0][b'hello'], HELLO)

    def test_wireFormatRoundTrip(self):
        """
        Verify that mixed-case, underscored and dashed arguments are mapped to
        their python names properly.
        """
        c, s, p = connectedServerAndClient()
        L = []
        HELLO = b'world'
        c.sendHello(HELLO).addCallback(L.append)
        p.flush()
        self.assertEqual(L[0][b'hello'], HELLO)

    def test_helloWorldUnicode(self):
        """
        Verify that unicode arguments can be encoded and decoded.
        """
        c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
        L = []
        HELLO = b'world'
        HELLO_UNICODE = 'worሴld'
        c.sendUnicodeHello(HELLO, HELLO_UNICODE).addCallback(L.append)
        p.flush()
        self.assertEqual(L[0]['hello'], HELLO)
        self.assertEqual(L[0]['Print'], HELLO_UNICODE)

    def test_callRemoteStringRequiresAnswerFalse(self):
        """
        L{BoxDispatcher.callRemoteString} returns L{None} if C{requiresAnswer}
        is C{False}.
        """
        c, s, p = connectedServerAndClient()
        ret = c.callRemoteString(b'WTF', requiresAnswer=False)
        self.assertIsNone(ret)

    def test_unknownCommandLow(self):
        """
        Verify that unknown commands using low-level APIs will be rejected with an
        error, but will NOT terminate the connection.
        """
        c, s, p = connectedServerAndClient()
        L = []

        def clearAndAdd(e):
            """
            You can't propagate the error...
            """
            e.trap(amp.UnhandledCommand)
            return 'OK'
        c.callRemoteString(b'WTF').addErrback(clearAndAdd).addCallback(L.append)
        p.flush()
        self.assertEqual(L.pop(), 'OK')
        HELLO = b'world'
        c.sendHello(HELLO).addCallback(L.append)
        p.flush()
        self.assertEqual(L[0][b'hello'], HELLO)

    def test_unknownCommandHigh(self):
        """
        Verify that unknown commands using high-level APIs will be rejected with an
        error, but will NOT terminate the connection.
        """
        c, s, p = connectedServerAndClient()
        L = []

        def clearAndAdd(e):
            """
            You can't propagate the error...
            """
            e.trap(amp.UnhandledCommand)
            return 'OK'
        c.callRemote(WTF).addErrback(clearAndAdd).addCallback(L.append)
        p.flush()
        self.assertEqual(L.pop(), 'OK')
        HELLO = b'world'
        c.sendHello(HELLO).addCallback(L.append)
        p.flush()
        self.assertEqual(L[0][b'hello'], HELLO)

    def test_brokenReturnValue(self):
        """
        It can be very confusing if you write some code which responds to a
        command, but gets the return value wrong.  Most commonly you end up
        returning None instead of a dictionary.

        Verify that if that happens, the framework logs a useful error.
        """
        L = []
        SimpleSymmetricCommandProtocol().dispatchCommand(amp.AmpBox(_command=BrokenReturn.commandName)).addErrback(L.append)
        L[0].trap(amp.BadLocalReturn)
        self.failUnlessIn('None', repr(L[0].value))

    def test_unknownArgument(self):
        """
        Verify that unknown arguments are ignored, and not passed to a Python
        function which can't accept them.
        """
        c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
        L = []
        HELLO = b'world'
        c.callRemote(FutureHello, hello=HELLO, bonus=b"I'm not in the book!").addCallback(L.append)
        p.flush()
        self.assertEqual(L[0]['hello'], HELLO)

    def test_simpleReprs(self):
        """
        Verify that the various Box objects repr properly, for debugging.
        """
        self.assertEqual(type(repr(amp._SwitchBox('a'))), str)
        self.assertEqual(type(repr(amp.QuitBox())), str)
        self.assertEqual(type(repr(amp.AmpBox())), str)
        self.assertIn('AmpBox', repr(amp.AmpBox()))

    def test_innerProtocolInRepr(self):
        """
        Verify that L{AMP} objects output their innerProtocol when set.
        """
        otherProto = TestProto(None, b'outgoing data')
        a = amp.AMP()
        a.innerProtocol = otherProto
        self.assertEqual(repr(a), '<AMP inner <TestProto #%d> at 0x%x>' % (otherProto.instanceId, id(a)))

    def test_innerProtocolNotInRepr(self):
        """
        Verify that L{AMP} objects do not output 'inner' when no innerProtocol
        is set.
        """
        a = amp.AMP()
        self.assertEqual(repr(a), f'<AMP at 0x{id(a):x}>')

    @skipIf(skipSSL, 'SSL not available')
    def test_simpleSSLRepr(self):
        """
        L{amp._TLSBox.__repr__} returns a string.
        """
        self.assertEqual(type(repr(amp._TLSBox())), str)

    def test_keyTooLong(self):
        """
        Verify that a key that is too long will immediately raise a synchronous
        exception.
        """
        c, s, p = connectedServerAndClient()
        x = 'H' * (255 + 1)
        tl = self.assertRaises(amp.TooLong, c.callRemoteString, b'Hello', **{x: b'hi'})
        self.assertTrue(tl.isKey)
        self.assertTrue(tl.isLocal)
        self.assertIsNone(tl.keyName)
        self.assertEqual(tl.value, x.encode('ascii'))
        self.assertIn(str(len(x)), repr(tl))
        self.assertIn('key', repr(tl))

    def test_valueTooLong(self):
        """
        Verify that attempting to send value longer than 64k will immediately
        raise an exception.
        """
        c, s, p = connectedServerAndClient()
        x = b'H' * (65535 + 1)
        tl = self.assertRaises(amp.TooLong, c.sendHello, x)
        p.flush()
        self.assertFalse(tl.isKey)
        self.assertTrue(tl.isLocal)
        self.assertEqual(tl.keyName, b'hello')
        self.failUnlessIdentical(tl.value, x)
        self.assertIn(str(len(x)), repr(tl))
        self.assertIn('value', repr(tl))
        self.assertIn('hello', repr(tl))

    def test_helloWorldCommand(self):
        """
        Verify that a simple command can be sent and its response received with
        the high-level value parsing API.
        """
        c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
        L = []
        HELLO = b'world'
        c.sendHello(HELLO).addCallback(L.append)
        p.flush()
        self.assertEqual(L[0]['hello'], HELLO)

    def test_helloErrorHandling(self):
        """
        Verify that if a known error type is raised and handled, it will be
        properly relayed to the other end of the connection and translated into
        an exception, and no error will be logged.
        """
        L = []
        c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
        HELLO = b'fuck you'
        c.sendHello(HELLO).addErrback(L.append)
        p.flush()
        L[0].trap(UnfriendlyGreeting)
        self.assertEqual(str(L[0].value), "Don't be a dick.")

    def test_helloFatalErrorHandling(self):
        """
        Verify that if a known, fatal error type is raised and handled, it will
        be properly relayed to the other end of the connection and translated
        into an exception, no error will be logged, and the connection will be
        terminated.
        """
        L = []
        c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
        HELLO = b'die'
        c.sendHello(HELLO).addErrback(L.append)
        p.flush()
        L.pop().trap(DeathThreat)
        c.sendHello(HELLO).addErrback(L.append)
        p.flush()
        L.pop().trap(error.ConnectionDone)

    def test_helloNoErrorHandling(self):
        """
        Verify that if an unknown error type is raised, it will be relayed to
        the other end of the connection and translated into an exception, it
        will be logged, and then the connection will be dropped.
        """
        L = []
        c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
        HELLO = THING_I_DONT_UNDERSTAND
        c.sendHello(HELLO).addErrback(L.append)
        p.flush()
        ure = L.pop()
        ure.trap(amp.UnknownRemoteError)
        c.sendHello(HELLO).addErrback(L.append)
        cl = L.pop()
        cl.trap(error.ConnectionDone)
        self.assertTrue(self.flushLoggedErrors(ThingIDontUnderstandError))

    def test_lateAnswer(self):
        """
        Verify that a command that does not get answered until after the
        connection terminates will not cause any errors.
        """
        c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
        L = []
        c.callRemote(WaitForever).addErrback(L.append)
        p.flush()
        self.assertEqual(L, [])
        s.transport.loseConnection()
        p.flush()
        L.pop().trap(error.ConnectionDone)
        s.waiting.callback({})
        return s.waiting

    def test_requiresNoAnswer(self):
        """
        Verify that a command that requires no answer is run.
        """
        c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
        HELLO = b'world'
        c.callRemote(NoAnswerHello, hello=HELLO)
        p.flush()
        self.assertTrue(s.greeted)

    def test_requiresNoAnswerFail(self):
        """
        Verify that commands sent after a failed no-answer request do not complete.
        """
        L = []
        c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
        HELLO = b'fuck you'
        c.callRemote(NoAnswerHello, hello=HELLO)
        p.flush()
        self.assertTrue(self.flushLoggedErrors(amp.RemoteAmpError))
        HELLO = b'world'
        c.callRemote(Hello, hello=HELLO).addErrback(L.append)
        p.flush()
        L.pop().trap(error.ConnectionDone)
        self.assertFalse(s.greeted)

    def test_requiresNoAnswerAfterFail(self):
        """
        No-answer commands sent after the connection has been torn down do not
        return a L{Deferred}.
        """
        c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
        c.transport.loseConnection()
        p.flush()
        result = c.callRemote(NoAnswerHello, hello=b'ignored')
        self.assertIs(result, None)

    def test_noAnswerResponderBadAnswer(self):
        """
        Verify that responders of requiresAnswer=False commands have to return
        a dictionary anyway.

        (requiresAnswer is a hint from the _client_ - the server may be called
        upon to answer commands in any case, if the client wants to know when
        they complete.)
        """
        c, s, p = connectedServerAndClient(ServerClass=BadNoAnswerCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
        c.callRemote(NoAnswerHello, hello=b'hello')
        p.flush()
        le = self.flushLoggedErrors(amp.BadLocalReturn)
        self.assertEqual(len(le), 1)

    def test_noAnswerResponderAskedForAnswer(self):
        """
        Verify that responders with requiresAnswer=False will actually respond
        if the client sets requiresAnswer=True.  In other words, verify that
        requiresAnswer is a hint honored only by the client.
        """
        c, s, p = connectedServerAndClient(ServerClass=NoAnswerCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
        L = []
        c.callRemote(Hello, hello=b'Hello!').addCallback(L.append)
        p.flush()
        self.assertEqual(len(L), 1)
        self.assertEqual(L, [dict(hello=b'Hello!-noanswer', Print=None)])

    def test_ampListCommand(self):
        """
        Test encoding of an argument that uses the AmpList encoding.
        """
        c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
        L = []
        c.callRemote(GetList, length=10).addCallback(L.append)
        p.flush()
        values = L.pop().get('body')
        self.assertEqual(values, [{'x': 1}] * 10)

    def test_optionalAmpListOmitted(self):
        """
        Sending a command with an omitted AmpList argument that is
        designated as optional does not raise an InvalidSignature error.
        """
        c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
        L = []
        c.callRemote(DontRejectMe, magicWord='please').addCallback(L.append)
        p.flush()
        response = L.pop().get('response')
        self.assertEqual(response, 'list omitted')

    def test_optionalAmpListPresent(self):
        """
        Sanity check that optional AmpList arguments are processed normally.
        """
        c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
        L = []
        c.callRemote(DontRejectMe, magicWord='please', list=[{'name': 'foo'}]).addCallback(L.append)
        p.flush()
        response = L.pop().get('response')
        self.assertEqual(response, 'foo accepted')

    def test_failEarlyOnArgSending(self):
        """
        Verify that if we pass an invalid argument list (omitting an argument),
        an exception will be raised.
        """
        self.assertRaises(amp.InvalidSignature, Hello)

    def test_doubleProtocolSwitch(self):
        """
        As a debugging aid, a protocol system should raise a
        L{ProtocolSwitched} exception when asked to switch a protocol that is
        already switched.
        """
        serverDeferred = defer.Deferred()
        serverProto = SimpleSymmetricCommandProtocol(serverDeferred)
        clientDeferred = defer.Deferred()
        clientProto = SimpleSymmetricCommandProtocol(clientDeferred)
        c, s, p = connectedServerAndClient(ServerClass=lambda: serverProto, ClientClass=lambda: clientProto)

        def switched(result):
            self.assertRaises(amp.ProtocolSwitched, c.switchToTestProtocol)
            self.testSucceeded = True
        c.switchToTestProtocol().addCallback(switched)
        p.flush()
        self.assertTrue(self.testSucceeded)

    def test_protocolSwitch(self, switcher=SimpleSymmetricCommandProtocol, spuriousTraffic=False, spuriousError=False):
        """
        Verify that it is possible to switch to another protocol mid-connection and
        send data to it successfully.
        """
        self.testSucceeded = False
        serverDeferred = defer.Deferred()
        serverProto = switcher(serverDeferred)
        clientDeferred = defer.Deferred()
        clientProto = switcher(clientDeferred)
        c, s, p = connectedServerAndClient(ServerClass=lambda: serverProto, ClientClass=lambda: clientProto)
        if spuriousTraffic:
            wfdr = []
            c.callRemote(WaitForever).addErrback(wfdr.append)
        switchDeferred = c.switchToTestProtocol()
        if spuriousTraffic:
            self.assertRaises(amp.ProtocolSwitched, c.sendHello, b'world')

        def cbConnsLost(info):
            (serverSuccess, serverData), (clientSuccess, clientData) = info
            self.assertTrue(serverSuccess)
            self.assertTrue(clientSuccess)
            self.assertEqual(b''.join(serverData), SWITCH_CLIENT_DATA)
            self.assertEqual(b''.join(clientData), SWITCH_SERVER_DATA)
            self.testSucceeded = True

        def cbSwitch(proto):
            return defer.DeferredList([serverDeferred, clientDeferred]).addCallback(cbConnsLost)
        switchDeferred.addCallback(cbSwitch)
        p.flush()
        if serverProto.maybeLater is not None:
            serverProto.maybeLater.callback(serverProto.maybeLaterProto)
            p.flush()
        if spuriousTraffic:
            if spuriousError:
                s.waiting.errback(amp.RemoteAmpError(b'SPURIOUS', "Here's some traffic in the form of an error."))
            else:
                s.waiting.callback({})
            p.flush()
        c.transport.loseConnection()
        p.flush()
        self.assertTrue(self.testSucceeded)

    def test_protocolSwitchDeferred(self):
        """
        Verify that protocol-switching even works if the value returned from
        the command that does the switch is deferred.
        """
        return self.test_protocolSwitch(switcher=DeferredSymmetricCommandProtocol)

    def test_protocolSwitchFail(self, switcher=SimpleSymmetricCommandProtocol):
        """
        Verify that if we try to switch protocols and it fails, the connection
        stays up and we can go back to speaking AMP.
        """
        self.testSucceeded = False
        serverDeferred = defer.Deferred()
        serverProto = switcher(serverDeferred)
        clientDeferred = defer.Deferred()
        clientProto = switcher(clientDeferred)
        c, s, p = connectedServerAndClient(ServerClass=lambda: serverProto, ClientClass=lambda: clientProto)
        L = []
        c.switchToTestProtocol(fail=True).addErrback(L.append)
        p.flush()
        L.pop().trap(UnknownProtocol)
        self.assertFalse(self.testSucceeded)
        c.sendHello(b'world').addCallback(L.append)
        p.flush()
        self.assertEqual(L.pop()['hello'], b'world')

    def test_trafficAfterSwitch(self):
        """
        Verify that attempts to send traffic after a switch will not corrupt
        the nested protocol.
        """
        return self.test_protocolSwitch(spuriousTraffic=True)

    def test_errorAfterSwitch(self):
        """
        Returning an error after a protocol switch should record the underlying
        error.
        """
        return self.test_protocolSwitch(spuriousTraffic=True, spuriousError=True)

    def test_quitBoxQuits(self):
        """
        Verify that commands with a responseType of QuitBox will in fact
        terminate the connection.
        """
        c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
        L = []
        HELLO = b'world'
        GOODBYE = b'everyone'
        c.sendHello(HELLO).addCallback(L.append)
        p.flush()
        self.assertEqual(L.pop()['hello'], HELLO)
        c.callRemote(Goodbye).addCallback(L.append)
        p.flush()
        self.assertEqual(L.pop()['goodbye'], GOODBYE)
        c.sendHello(HELLO).addErrback(L.append)
        L.pop().trap(error.ConnectionDone)

    def test_basicLiteralEmit(self):
        """
        Verify that the command dictionaries for a callRemoteN look correct
        after being serialized and parsed.
        """
        c, s, p = connectedServerAndClient()
        L = []
        s.ampBoxReceived = L.append
        c.callRemote(Hello, hello=b'hello test', mixedCase=b'mixed case arg test', dash_arg=b'x', underscore_arg=b'y')
        p.flush()
        self.assertEqual(len(L), 1)
        for k, v in [(b'_command', Hello.commandName), (b'hello', b'hello test'), (b'mixedCase', b'mixed case arg test'), (b'dash-arg', b'x'), (b'underscore_arg', b'y')]:
            self.assertEqual(L[-1].pop(k), v)
        L[-1].pop(b'_ask')
        self.assertEqual(L[-1], {})

    def test_basicStructuredEmit(self):
        """
        Verify that a call similar to basicLiteralEmit's is handled properly with
        high-level quoting and passing to Python methods, and that argument
        names are correctly handled.
        """
        L = []

        class StructuredHello(amp.AMP):

            def h(self, *a, **k):
                L.append((a, k))
                return dict(hello=b'aaa')
            Hello.responder(h)
        c, s, p = connectedServerAndClient(ServerClass=StructuredHello)
        c.callRemote(Hello, hello=b'hello test', mixedCase=b'mixed case arg test', dash_arg=b'x', underscore_arg=b'y').addCallback(L.append)
        p.flush()
        self.assertEqual(len(L), 2)
        self.assertEqual(L[0], ((), dict(hello=b'hello test', mixedCase=b'mixed case arg test', dash_arg=b'x', underscore_arg=b'y', From=s.transport.getPeer(), Print=None, optional=None)))
        self.assertEqual(L[1], dict(Print=None, hello=b'aaa'))