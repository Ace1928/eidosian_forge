from zope.interface import implementer
from twisted.cred import error
from twisted.cred.credentials import Anonymous
from twisted.logger import Logger
from twisted.python.components import proxyForInterface
from twisted.web import util
from twisted.web.resource import IResource, _UnsafeErrorPage
def _authorizedResource(self, request):
    """
        Get the L{IResource} which the given request is authorized to receive.
        If the proper authorization headers are present, the resource will be
        requested from the portal.  If not, an anonymous login attempt will be
        made.
        """
    authheader = request.getHeader(b'authorization')
    if not authheader:
        return util.DeferredResource(self._login(Anonymous()))
    factory, respString = self._selectParseHeader(authheader)
    if factory is None:
        return UnauthorizedResource(self._credentialFactories)
    try:
        credentials = factory.decode(respString, request)
    except error.LoginFailed:
        return UnauthorizedResource(self._credentialFactories)
    except BaseException:
        self._log.failure('Unexpected failure from credentials factory')
        return _UnsafeErrorPage(500, 'Internal Error', '')
    else:
        return util.DeferredResource(self._login(credentials))