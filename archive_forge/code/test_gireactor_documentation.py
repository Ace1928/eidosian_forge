from __future__ import annotations
from unittest import skipIf
from twisted.internet.error import ReactorAlreadyRunning
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.trial.unittest import SkipTest, TestCase

        It is not possible to register more than one C{Application}.
        