from sys import intern
from typing import Type
from twisted.internet import protocol
from twisted.python import failure
from twisted.words.xish import domish, utility
class BootstrapMixin:
    """
    XmlStream factory mixin to install bootstrap event observers.

    This mixin is for factories providing
    L{IProtocolFactory<twisted.internet.interfaces.IProtocolFactory>} to make
    sure bootstrap event observers are set up on protocols, before incoming
    data is processed. Such protocols typically derive from
    L{utility.EventDispatcher}, like L{XmlStream}.

    You can set up bootstrap event observers using C{addBootstrap}. The
    C{event} and C{fn} parameters correspond with the C{event} and
    C{observerfn} arguments to L{utility.EventDispatcher.addObserver}.

    @since: 8.2.
    @ivar bootstraps: The list of registered bootstrap event observers.
    @type bootstrap: C{list}
    """

    def __init__(self):
        self.bootstraps = []

    def installBootstraps(self, dispatcher):
        """
        Install registered bootstrap observers.

        @param dispatcher: Event dispatcher to add the observers to.
        @type dispatcher: L{utility.EventDispatcher}
        """
        for event, fn in self.bootstraps:
            dispatcher.addObserver(event, fn)

    def addBootstrap(self, event, fn):
        """
        Add a bootstrap event handler.

        @param event: The event to register an observer for.
        @type event: C{str} or L{xpath.XPathQuery}
        @param fn: The observer callable to be registered.
        """
        self.bootstraps.append((event, fn))

    def removeBootstrap(self, event, fn):
        """
        Remove a bootstrap event handler.

        @param event: The event the observer is registered for.
        @type event: C{str} or L{xpath.XPathQuery}
        @param fn: The registered observer callable.
        """
        self.bootstraps.remove((event, fn))