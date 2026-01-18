from twisted.internet.error import ReactorAlreadyInstalledError
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest

        For backwards compatibility, L{ReactorAlreadyInstalledError} is an
        L{AssertionError}.
        