import sys
from typing import Optional, Sequence, Type
from zope.interface import Attribute, Interface
from twisted.plugin import getPlugins
from twisted.python import usage
class AuthOptionMixin:
    """
    Defines helper methods that can be added on to any
    L{usage.Options} subclass that needs authentication.

    This mixin implements three new options methods:

    The opt_auth method (--auth) will write two new values to the
    'self' dictionary: C{credInterfaces} (a dict of lists) and
    C{credCheckers} (a list).

    The opt_help_auth method (--help-auth) will search for all
    available checker plugins and list them for the user; it will exit
    when finished.

    The opt_help_auth_type method (--help-auth-type) will display
    detailed help for a particular checker plugin.

    @cvar supportedInterfaces: An iterable object that returns
       credential interfaces which this application is able to support.

    @cvar authOutput: A writeable object to which this options class
        will send all help-related output. Default: L{sys.stdout}
    """
    supportedInterfaces: Optional[Sequence[Type[Interface]]] = None
    authOutput = sys.stdout

    def supportsInterface(self, interface):
        """
        Returns whether a particular credentials interface is supported.
        """
        return self.supportedInterfaces is None or interface in self.supportedInterfaces

    def supportsCheckerFactory(self, factory):
        """
        Returns whether a checker factory will provide at least one of
        the credentials interfaces that we care about.
        """
        for interface in factory.credentialInterfaces:
            if self.supportsInterface(interface):
                return True
        return False

    def addChecker(self, checker):
        """
        Supply a supplied credentials checker to the Options class.
        """
        supported = []
        if self.supportedInterfaces is None:
            supported = checker.credentialInterfaces
        else:
            for interface in checker.credentialInterfaces:
                if self.supportsInterface(interface):
                    supported.append(interface)
        if not supported:
            raise UnsupportedInterfaces(checker.credentialInterfaces)
        if 'credInterfaces' not in self:
            self['credInterfaces'] = {}
        if 'credCheckers' not in self:
            self['credCheckers'] = []
        self['credCheckers'].append(checker)
        for interface in supported:
            self['credInterfaces'].setdefault(interface, []).append(checker)

    def opt_auth(self, description):
        """
        Specify an authentication method for the server.
        """
        try:
            self.addChecker(makeChecker(description))
        except UnsupportedInterfaces as e:
            raise usage.UsageError('Auth plugin not supported: %s' % e.args[0])
        except InvalidAuthType as e:
            raise usage.UsageError('Auth plugin not recognized: %s' % e.args[0])
        except Exception as e:
            raise usage.UsageError('Unexpected error: %s' % e)

    def _checkerFactoriesForOptHelpAuth(self):
        """
        Return a list of which authTypes will be displayed by --help-auth.
        This makes it a lot easier to test this module.
        """
        for factory in findCheckerFactories():
            for interface in factory.credentialInterfaces:
                if self.supportsInterface(interface):
                    yield factory
                    break

    def opt_help_auth(self):
        """
        Show all authentication methods available.
        """
        self.authOutput.write('Usage: --auth AuthType[:ArgString]\n')
        self.authOutput.write('For detailed help: --help-auth-type AuthType\n')
        self.authOutput.write('\n')
        firstLength = 0
        for factory in self._checkerFactoriesForOptHelpAuth():
            if len(factory.authType) > firstLength:
                firstLength = len(factory.authType)
        formatString = '  %%-%is\t%%s\n' % firstLength
        self.authOutput.write(formatString % ('AuthType', 'ArgString format'))
        self.authOutput.write(formatString % ('========', '================'))
        for factory in self._checkerFactoriesForOptHelpAuth():
            self.authOutput.write(formatString % (factory.authType, factory.argStringFormat))
        self.authOutput.write('\n')
        raise SystemExit(0)

    def opt_help_auth_type(self, authType):
        """
        Show help for a particular authentication type.
        """
        try:
            cf = findCheckerFactory(authType)
        except InvalidAuthType:
            raise usage.UsageError('Invalid auth type: %s' % authType)
        self.authOutput.write('Usage: --auth %s[:ArgString]\n' % authType)
        self.authOutput.write('ArgString format: %s\n' % cf.argStringFormat)
        self.authOutput.write('\n')
        for line in cf.authHelp.strip().splitlines():
            self.authOutput.write('  %s\n' % line.rstrip())
        self.authOutput.write('\n')
        if not self.supportsCheckerFactory(cf):
            self.authOutput.write('  %s\n' % notSupportedWarning)
            self.authOutput.write('\n')
        raise SystemExit(0)