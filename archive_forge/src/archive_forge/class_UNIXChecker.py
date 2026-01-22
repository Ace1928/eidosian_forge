from zope.interface import implementer
from twisted import plugin
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import IUsernamePassword
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.strcred import ICheckerFactory
from twisted.internet import defer
@implementer(ICredentialsChecker)
class UNIXChecker:
    """
    A credentials checker for a UNIX server. This will check that
    an authenticating username/password is a valid user on the system.

    Does not work on Windows.

    Right now this supports Python's pwd and spwd modules, if they are
    installed. It does not support PAM.
    """
    credentialInterfaces = (IUsernamePassword,)

    def checkPwd(self, pwd, username, password):
        """
        Obtain the encrypted password for C{username} from the Unix password
        database using L{pwd.getpwnam}, and see if it it matches it matches
        C{password}.

        @param pwd: Module which provides functions which
                    access to the Unix password database.
        @type pwd: C{module}
        @param username: The user to look up in the Unix password database.
        @type username: L{unicode}/L{str} or L{bytes}
        @param password: The password to compare.
        @type username: L{unicode}/L{str} or L{bytes}
        """
        try:
            if isinstance(username, bytes):
                username = username.decode('utf-8')
            cryptedPass = pwd.getpwnam(username).pw_passwd
        except KeyError:
            return defer.fail(UnauthorizedLogin())
        else:
            if cryptedPass in ('*', 'x'):
                return None
            elif verifyCryptedPassword(cryptedPass, password):
                return defer.succeed(username)

    def checkSpwd(self, spwd, username, password):
        """
        Obtain the encrypted password for C{username} from the
        Unix shadow password database using L{spwd.getspnam},
        and see if it it matches it matches C{password}.

        @param spwd: Module which provides functions which
                     access to the Unix shadow password database.
        @type spwd: C{module}
        @param username: The user to look up in the Unix password database.
        @type username: L{unicode}/L{str} or L{bytes}
        @param password: The password to compare.
        @type username: L{unicode}/L{str} or L{bytes}
        """
        try:
            if isinstance(username, bytes):
                username = username.decode('utf-8')
            if getattr(spwd.struct_spwd, 'sp_pwdp', None):
                cryptedPass = spwd.getspnam(username).sp_pwdp
            else:
                cryptedPass = spwd.getspnam(username).sp_pwd
        except KeyError:
            return defer.fail(UnauthorizedLogin())
        else:
            if verifyCryptedPassword(cryptedPass, password):
                return defer.succeed(username)

    def requestAvatarId(self, credentials):
        username, password = (credentials.username, credentials.password)
        try:
            import pwd
        except ImportError:
            pwd = None
        if pwd is not None:
            checked = self.checkPwd(pwd, username, password)
            if checked is not None:
                return checked
        try:
            import spwd
        except ImportError:
            spwd = None
        if spwd is not None:
            checked = self.checkSpwd(spwd, username, password)
            if checked is not None:
                return checked
        return defer.fail(UnauthorizedLogin())