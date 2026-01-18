from types import ModuleType
from typing import Optional
from zope.interface import implementer
from twisted.conch.error import ConchError, ValidPublicKey
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import IAnonymous, ISSHPrivateKey, IUsernamePassword
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import defer, task
from twisted.protocols import loopback
from twisted.python.reflect import requireModule
from twisted.trial import unittest

            Return our fake service.
            