import string
class MethodInjectionTestsMixin:
    """
    A mixin that runs HTTP method injection tests.  Define
    L{MethodInjectionTestsMixin.attemptRequestWithMaliciousMethod} in
    a L{twisted.trial.unittest.SynchronousTestCase} subclass to test
    how HTTP client code behaves when presented with malicious HTTP
    methods.

    @see: U{CVE-2019-12387}
    """

    def attemptRequestWithMaliciousMethod(self, method):
        """
        Attempt to send a request with the given method.  This should
        synchronously raise a L{ValueError} if either is invalid.

        @param method: the method (e.g. C{GET\x00})

        @param uri: the URI

        @type method:
        """
        raise NotImplementedError()

    def test_methodWithCLRFRejected(self):
        """
        Issuing a request with a method that contains a carriage
        return and line feed fails with a L{ValueError}.
        """
        with self.assertRaises(ValueError) as cm:
            method = b'GET\r\nX-Injected-Header: value'
            self.attemptRequestWithMaliciousMethod(method)
        self.assertRegex(str(cm.exception), '^Invalid method')

    def test_methodWithUnprintableASCIIRejected(self):
        """
        Issuing a request with a method that contains unprintable
        ASCII characters fails with a L{ValueError}.
        """
        for c in UNPRINTABLE_ASCII:
            method = b'GET%s' % (bytearray([c]),)
            with self.assertRaises(ValueError) as cm:
                self.attemptRequestWithMaliciousMethod(method)
            self.assertRegex(str(cm.exception), '^Invalid method')

    def test_methodWithNonASCIIRejected(self):
        """
        Issuing a request with a method that contains non-ASCII
        characters fails with a L{ValueError}.
        """
        for c in NONASCII:
            method = b'GET%s' % (bytearray([c]),)
            with self.assertRaises(ValueError) as cm:
                self.attemptRequestWithMaliciousMethod(method)
            self.assertRegex(str(cm.exception), '^Invalid method')