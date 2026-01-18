from __future__ import annotations
from twisted.cred import credentials, error
from twisted.cred.checkers import FilePasswordDB
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words import tap

        This method executes both positive and negative authentication
        tests against whatever credentials checker has been stored in
        the Options class.

        @param opt: An instance of L{tap.Options}.
        