import time
from twisted.internet import defer
from twisted.spread import banana, flavors, jelly
def _cbGotUpdate(self, newState):
    self.__dict__.update(newState)
    self.isActivated = 1
    for listener in self._activationListeners:
        listener(self)
    self._activationListeners = []
    self.activated()
    with open(self.getFileName(), 'wb') as dataFile:
        dataFile.write(banana.encode(jelly.jelly(self)))