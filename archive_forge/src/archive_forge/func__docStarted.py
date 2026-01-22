from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def _docStarted(self, root):
    self.root = root
    self.doc_started = True