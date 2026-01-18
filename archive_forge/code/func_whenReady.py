import time
from twisted.internet import defer
from twisted.spread import banana, flavors, jelly
def whenReady(d):
    """
    Wrap a deferred returned from a pb method in another deferred that
    expects a RemotePublished as a result.  This will allow you to wait until
    the result is really available.

    Idiomatic usage would look like::

        publish.whenReady(serverObject.getMeAPublishable()).addCallback(lookAtThePublishable)
    """
    d2 = defer.Deferred()
    d.addCallbacks(_pubReady, d2.errback, callbackArgs=(d2,))
    return d2