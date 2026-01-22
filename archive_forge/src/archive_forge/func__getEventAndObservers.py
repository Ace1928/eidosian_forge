from twisted.python import log
from twisted.words.xish import xpath
def _getEventAndObservers(self, event):
    if isinstance(event, xpath.XPathQuery):
        observers = self._xpathObservers
    elif self.prefix == event[:len(self.prefix)]:
        observers = self._eventObservers
    else:
        event = xpath.internQuery(event)
        observers = self._xpathObservers
    return (event, observers)