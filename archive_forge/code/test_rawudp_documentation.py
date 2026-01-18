from __future__ import annotations
from twisted.internet import protocol
from twisted.pair import rawudp
from twisted.trial import unittest
Adding a protocol with a number >=2**16 raises an exception.