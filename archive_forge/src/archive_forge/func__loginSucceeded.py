from zope.interface import implementer
from twisted.cred import error
from twisted.cred.credentials import Anonymous
from twisted.logger import Logger
from twisted.python.components import proxyForInterface
from twisted.web import util
from twisted.web.resource import IResource, _UnsafeErrorPage
def _loginSucceeded(self, args):
    """
        Handle login success by wrapping the resulting L{IResource} avatar
        so that the C{logout} callback will be invoked when rendering is
        complete.
        """
    interface, avatar, logout = args

    class ResourceWrapper(proxyForInterface(IResource, 'resource')):
        """
            Wrap an L{IResource} so that whenever it or a child of it
            completes rendering, the cred logout hook will be invoked.

            An assumption is made here that exactly one L{IResource} from
            among C{avatar} and all of its children will be rendered.  If
            more than one is rendered, C{logout} will be invoked multiple
            times and probably earlier than desired.
            """

        def getChildWithDefault(self, name, request):
            """
                Pass through the lookup to the wrapped resource, wrapping
                the result in L{ResourceWrapper} to ensure C{logout} is
                called when rendering of the child is complete.
                """
            return ResourceWrapper(self.resource.getChildWithDefault(name, request))

        def render(self, request):
            """
                Hook into response generation so that when rendering has
                finished completely (with or without error), C{logout} is
                called.
                """
            request.notifyFinish().addBoth(lambda ign: logout())
            return super().render(request)
    return ResourceWrapper(avatar)