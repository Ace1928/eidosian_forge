import sys
from typing import Optional, Sequence, Type
from zope.interface import Attribute, Interface
from twisted.plugin import getPlugins
from twisted.python import usage
class ICheckerFactory(Interface):
    """
    A factory for objects which provide
    L{twisted.cred.checkers.ICredentialsChecker}.

    It's implemented by twistd plugins creating checkers.
    """
    authType = Attribute('A tag that identifies the authentication method.')
    authHelp = Attribute('A detailed (potentially multi-line) description of precisely what functionality this CheckerFactory provides.')
    argStringFormat = Attribute('A short (one-line) description of the argument string format.')
    credentialInterfaces = Attribute('A list of credentials interfaces that this factory will support.')

    def generateChecker(argstring):
        """
        Return an L{twisted.cred.checkers.ICredentialsChecker} provider using the supplied
        argument string.
        """