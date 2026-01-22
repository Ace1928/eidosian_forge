from zope.interface import implementer
from twisted.cred import error
from twisted.cred.credentials import Anonymous
from twisted.logger import Logger
from twisted.python.components import proxyForInterface
from twisted.web import util
from twisted.web.resource import IResource, _UnsafeErrorPage
def _selectParseHeader(self, header):
    """
        Choose an C{ICredentialFactory} from C{_credentialFactories}
        suitable to use to decode the given I{Authenticate} header.

        @return: A two-tuple of a factory and the remaining portion of the
            header value to be decoded or a two-tuple of L{None} if no
            factory can decode the header value.
        """
    elements = header.split(b' ')
    scheme = elements[0].lower()
    for fact in self._credentialFactories:
        if fact.scheme == scheme:
            return (fact, b' '.join(elements[1:]))
    return (None, None)