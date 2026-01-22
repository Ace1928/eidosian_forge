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
def _shadowGetByName(username: str) -> Optional[CryptedPasswordRecord]:
    """
    Look up a user in the /etc/shadow database using the spwd module. If it is
    not available, return L{None}.

    @param username: the username of the user to return the shadow database
        information for.
    @type username: L{str}

    @returns: A L{spwd.struct_spwd}, where field 1 may contain a crypted
        password, or L{None} when the L{spwd} database is unavailable.

    @raises KeyError: when no such user exists
    """
    if spwd is not None:
        f = spwd.getspnam
    else:
        return None
    return cast(CryptedPasswordRecord, runAsEffectiveUser(0, 0, f, username))