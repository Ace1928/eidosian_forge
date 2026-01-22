import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def _testPublicPrivateFromString(self, public, private, type, data):
    self._testPublicFromString(public, type, data)
    self._testPrivateFromString(private, type, data)