from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
class PBFailureTests(PBConnTestCase):
    compare = unittest.TestCase.assertEqual

    def _exceptionTest(self, method, exceptionType, flush):

        def eb(err):
            err.trap(exceptionType)
            self.compare(err.traceback, 'Traceback unavailable\n')
            if flush:
                errs = self.flushLoggedErrors(exceptionType)
                self.assertEqual(len(errs), 1)
            return (err.type, err.value, err.traceback)
        d = self.clientFactory.getRootObject()

        def gotRootObject(root):
            d = root.callRemote(method)
            d.addErrback(eb)
            return d
        d.addCallback(gotRootObject)
        self.pump.flush()

    def test_asynchronousException(self):
        """
        Test that a Deferred returned by a remote method which already has a
        Failure correctly has that error passed back to the calling side.
        """
        return self._exceptionTest('asynchronousException', AsynchronousException, True)

    def test_synchronousException(self):
        """
        Like L{test_asynchronousException}, but for a method which raises an
        exception synchronously.
        """
        return self._exceptionTest('synchronousException', SynchronousException, True)

    def test_asynchronousError(self):
        """
        Like L{test_asynchronousException}, but for a method which returns a
        Deferred failing with an L{pb.Error} subclass.
        """
        return self._exceptionTest('asynchronousError', AsynchronousError, False)

    def test_synchronousError(self):
        """
        Like L{test_asynchronousError}, but for a method which synchronously
        raises a L{pb.Error} subclass.
        """
        return self._exceptionTest('synchronousError', SynchronousError, False)

    def _success(self, result, expectedResult):
        self.assertEqual(result, expectedResult)
        return result

    def _addFailingCallbacks(self, remoteCall, expectedResult, eb):
        remoteCall.addCallbacks(self._success, eb, callbackArgs=(expectedResult,))
        return remoteCall

    def _testImpl(self, method, expected, eb, exc=None):
        """
        Call the given remote method and attach the given errback to the
        resulting Deferred.  If C{exc} is not None, also assert that one
        exception of that type was logged.
        """
        rootDeferred = self.clientFactory.getRootObject()

        def gotRootObj(obj):
            failureDeferred = self._addFailingCallbacks(obj.callRemote(method), expected, eb)
            if exc is not None:

                def gotFailure(err):
                    self.assertEqual(len(self.flushLoggedErrors(exc)), 1)
                    return err
                failureDeferred.addBoth(gotFailure)
            return failureDeferred
        rootDeferred.addCallback(gotRootObj)
        self.pump.flush()

    def test_jellyFailure(self):
        """
        Test that an exception which is a subclass of L{pb.Error} has more
        information passed across the network to the calling side.
        """

        def failureJelly(fail):
            fail.trap(JellyError)
            self.assertNotIsInstance(fail.type, str)
            self.assertIsInstance(fail.value, fail.type)
            return 43
        return self._testImpl('jelly', 43, failureJelly)

    def test_deferredJellyFailure(self):
        """
        Test that a Deferred which fails with a L{pb.Error} is treated in
        the same way as a synchronously raised L{pb.Error}.
        """

        def failureDeferredJelly(fail):
            fail.trap(JellyError)
            self.assertNotIsInstance(fail.type, str)
            self.assertIsInstance(fail.value, fail.type)
            return 430
        return self._testImpl('deferredJelly', 430, failureDeferredJelly)

    def test_unjellyableFailure(self):
        """
        A non-jellyable L{pb.Error} subclass raised by a remote method is
        turned into a Failure with a type set to the FQPN of the exception
        type.
        """

        def failureUnjellyable(fail):
            self.assertEqual(fail.type, b'twisted.spread.test.test_pbfailure.SynchronousError')
            return 431
        return self._testImpl('synchronousError', 431, failureUnjellyable)

    def test_unknownFailure(self):
        """
        Test that an exception which is a subclass of L{pb.Error} but not
        known on the client side has its type set properly.
        """

        def failureUnknown(fail):
            self.assertEqual(fail.type, b'twisted.spread.test.test_pbfailure.UnknownError')
            return 4310
        return self._testImpl('unknownError', 4310, failureUnknown)

    def test_securityFailure(self):
        """
        Test that even if an exception is not explicitly jellyable (by being
        a L{pb.Jellyable} subclass), as long as it is an L{pb.Error}
        subclass it receives the same special treatment.
        """

        def failureSecurity(fail):
            fail.trap(SecurityError)
            self.assertNotIsInstance(fail.type, str)
            self.assertIsInstance(fail.value, fail.type)
            return 4300
        return self._testImpl('security', 4300, failureSecurity)

    def test_deferredSecurity(self):
        """
        Test that a Deferred which fails with a L{pb.Error} which is not
        also a L{pb.Jellyable} is treated in the same way as a synchronously
        raised exception of the same type.
        """

        def failureDeferredSecurity(fail):
            fail.trap(SecurityError)
            self.assertNotIsInstance(fail.type, str)
            self.assertIsInstance(fail.value, fail.type)
            return 43000
        return self._testImpl('deferredSecurity', 43000, failureDeferredSecurity)

    def test_noSuchMethodFailure(self):
        """
        Test that attempting to call a method which is not defined correctly
        results in an AttributeError on the calling side.
        """

        def failureNoSuch(fail):
            fail.trap(pb.NoSuchMethod)
            self.compare(fail.traceback, 'Traceback unavailable\n')
            return 42000
        return self._testImpl('nosuch', 42000, failureNoSuch, AttributeError)

    def test_copiedFailureLogging(self):
        """
        Test that a copied failure received from a PB call can be logged
        locally.

        Note: this test needs some serious help: all it really tests is that
        log.err(copiedFailure) doesn't raise an exception.
        """
        d = self.clientFactory.getRootObject()

        def connected(rootObj):
            return rootObj.callRemote('synchronousException')
        d.addCallback(connected)

        def exception(failure):
            log.err(failure)
            errs = self.flushLoggedErrors(SynchronousException)
            self.assertEqual(len(errs), 2)
        d.addErrback(exception)
        self.pump.flush()

    def test_throwExceptionIntoGenerator(self):
        """
        L{pb.CopiedFailure.throwExceptionIntoGenerator} will throw a
        L{RemoteError} into the given paused generator at the point where it
        last yielded.
        """
        original = pb.CopyableFailure(AttributeError('foo'))
        copy = jelly.unjelly(jelly.jelly(original, invoker=DummyInvoker()))
        exception = []

        def generatorFunc():
            try:
                yield None
            except pb.RemoteError as exc:
                exception.append(exc)
            else:
                self.fail('RemoteError not raised')
        gen = generatorFunc()
        gen.send(None)
        self.assertRaises(StopIteration, copy.throwExceptionIntoGenerator, gen)
        self.assertEqual(len(exception), 1)
        exc = exception[0]
        self.assertEqual(exc.remoteType, qual(AttributeError).encode('ascii'))
        self.assertEqual(exc.args, ('foo',))
        self.assertEqual(exc.remoteTraceback, 'Traceback unavailable\n')