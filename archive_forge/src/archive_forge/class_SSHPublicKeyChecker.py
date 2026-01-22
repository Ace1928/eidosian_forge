import binascii
import errno
import sys
from base64 import decodebytes
from typing import IO, Any, Callable, Iterable, Iterator, Mapping, Optional, Tuple, cast
from zope.interface import Interface, implementer, providedBy
from incremental import Version
from typing_extensions import Literal, Protocol
from twisted.conch import error
from twisted.conch.ssh import keys
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import ISSHPrivateKey, IUsernamePassword
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import defer
from twisted.logger import Logger
from twisted.plugins.cred_unix import verifyCryptedPassword
from twisted.python import failure, reflect
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.filepath import FilePath
from twisted.python.util import runAsEffectiveUser
@implementer(ICredentialsChecker)
class SSHPublicKeyChecker:
    """
    Checker that authenticates SSH public keys, based on public keys listed in
    authorized_keys and authorized_keys2 files in user .ssh/ directories.

    Initializing this checker with a L{UNIXAuthorizedKeysFiles} should be
    used instead of L{twisted.conch.checkers.SSHPublicKeyDatabase}.

    @since: 15.0
    """
    credentialInterfaces = (ISSHPrivateKey,)

    def __init__(self, keydb: IAuthorizedKeysDB) -> None:
        """
        Initializes a L{SSHPublicKeyChecker}.

        @param keydb: a provider of L{IAuthorizedKeysDB}
        """
        self._keydb = keydb

    def requestAvatarId(self, credentials):
        d = defer.execute(self._sanityCheckKey, credentials)
        d.addCallback(self._checkKey, credentials)
        d.addCallback(self._verifyKey, credentials)
        return d

    def _sanityCheckKey(self, credentials):
        """
        Checks whether the provided credentials are a valid SSH key with a
        signature (does not actually verify the signature).

        @param credentials: the credentials offered by the user
        @type credentials: L{ISSHPrivateKey} provider

        @raise ValidPublicKey: the credentials do not include a signature. See
            L{error.ValidPublicKey} for more information.

        @raise BadKeyError: The key included with the credentials is not
            recognized as a key.

        @return: the key in the credentials
        @rtype: L{twisted.conch.ssh.keys.Key}
        """
        if not credentials.signature:
            raise error.ValidPublicKey()
        return keys.Key.fromString(credentials.blob)

    def _checkKey(self, pubKey, credentials):
        """
        Checks the public key against all authorized keys (if any) for the
        user.

        @param pubKey: the key in the credentials (just to prevent it from
            having to be calculated again)
        @type pubKey:

        @param credentials: the credentials offered by the user
        @type credentials: L{ISSHPrivateKey} provider

        @raise UnauthorizedLogin: If the key is not authorized, or if there
            was any error obtaining a list of authorized keys for the user.

        @return: C{pubKey} if the key is authorized
        @rtype: L{twisted.conch.ssh.keys.Key}
        """
        if any((key == pubKey for key in self._keydb.getAuthorizedKeys(credentials.username))):
            return pubKey
        raise UnauthorizedLogin('Key not authorized')

    def _verifyKey(self, pubKey, credentials):
        """
        Checks whether the credentials themselves are valid, now that we know
        if the key matches the user.

        @param pubKey: the key in the credentials (just to prevent it from
            having to be calculated again)
        @type pubKey: L{twisted.conch.ssh.keys.Key}

        @param credentials: the credentials offered by the user
        @type credentials: L{ISSHPrivateKey} provider

        @raise UnauthorizedLogin: If the key signature is invalid or there
            was any error verifying the signature.

        @return: The user's username, if authentication was successful
        @rtype: L{bytes}
        """
        try:
            if pubKey.verify(credentials.signature, credentials.sigData):
                return credentials.username
        except Exception as e:
            raise UnauthorizedLogin('Error while verifying key') from e
        raise UnauthorizedLogin('Key signature invalid.')