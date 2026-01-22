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
class IAuthorizedKeysDB(Interface):
    """
    An object that provides valid authorized ssh keys mapped to usernames.

    @since: 15.0
    """

    def getAuthorizedKeys(avatarId):
        """
        Gets an iterable of authorized keys that are valid for the given
        C{avatarId}.

        @param avatarId: the ID of the avatar
        @type avatarId: valid return value of
            L{twisted.cred.checkers.ICredentialsChecker.requestAvatarId}

        @return: an iterable of L{twisted.conch.ssh.keys.Key}
        """