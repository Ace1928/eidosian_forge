from sys import intern
from typing import Type
from twisted.internet import protocol
from twisted.python import failure
from twisted.words.xish import domish, utility
def _initializeStream(self):
    """Sets up XML Parser."""
    self.stream = domish.elementStream()
    self.stream.DocumentStartEvent = self.onDocumentStart
    self.stream.ElementEvent = self.onElement
    self.stream.DocumentEndEvent = self.onDocumentEnd