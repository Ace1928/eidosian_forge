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
@implementer(IAuthorizedKeysDB)
class InMemorySSHKeyDB:
    """
    Object that provides SSH public keys based on a dictionary of usernames
    mapped to L{twisted.conch.ssh.keys.Key}s.

    @since: 15.0
    """

    def __init__(self, mapping: Mapping[bytes, Iterable[keys.Key]]) -> None:
        """
        Initializes a new L{InMemorySSHKeyDB}.

        @param mapping: mapping of usernames to iterables of
            L{twisted.conch.ssh.keys.Key}s

        """
        self._mapping = mapping

    def getAuthorizedKeys(self, username: bytes) -> Iterable[keys.Key]:
        """
        Look up the authorized keys for a user.

        @param username: Name of the user
        """
        return self._mapping.get(username, [])