import time
from twisted.internet import defer
from twisted.spread import banana, flavors, jelly
def view_getStateToPublish(self, perspective):
    """(internal)"""
    return self.getStateToPublishFor(perspective)