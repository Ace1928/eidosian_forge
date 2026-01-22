from zope.interface import implementer
from twisted import plugin
from twisted.cred.checkers import AllowAnonymousAccess
from twisted.cred.credentials import IAnonymous
from twisted.cred.strcred import ICheckerFactory
@implementer(ICheckerFactory, plugin.IPlugin)
class AnonymousCheckerFactory:
    """
    Generates checkers that will authenticate an anonymous request.
    """
    authType = 'anonymous'
    authHelp = anonymousCheckerFactoryHelp
    argStringFormat = 'No argstring required.'
    credentialInterfaces = (IAnonymous,)

    def generateChecker(self, argstring=''):
        return AllowAnonymousAccess()