from twisted.words.protocols.jabber import error, sasl, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish, utility, xpath
class IQ(domish.Element):
    """
    Wrapper for a Info/Query packet.

    This provides the necessary functionality to send IQs and get notified when
    a result comes back. It's a subclass from L{domish.Element}, so you can use
    the standard DOM manipulation calls to add data to the outbound request.

    @type callbacks: L{utility.CallbackList}
    @cvar callbacks: Callback list to be notified when response comes back

    """

    def __init__(self, xmlstream, type='set'):
        """
        @type xmlstream: L{xmlstream.XmlStream}
        @param xmlstream: XmlStream to use for transmission of this IQ

        @type type: C{str}
        @param type: IQ type identifier ('get' or 'set')
        """
        domish.Element.__init__(self, ('jabber:client', 'iq'))
        self.addUniqueId()
        self['type'] = type
        self._xmlstream = xmlstream
        self.callbacks = utility.CallbackList()

    def addCallback(self, fn, *args, **kwargs):
        """
        Register a callback for notification when the IQ result is available.
        """
        self.callbacks.addCallback(True, fn, *args, **kwargs)

    def send(self, to=None):
        """
        Call this method to send this IQ request via the associated XmlStream.

        @param to: Jabber ID of the entity to send the request to
        @type to: C{str}

        @returns: Callback list for this IQ. Any callbacks added to this list
                  will be fired when the result comes back.
        """
        if to != None:
            self['to'] = to
        self._xmlstream.addOnetimeObserver("/iq[@id='%s']" % self['id'], self._resultEvent)
        self._xmlstream.send(self)

    def _resultEvent(self, iq):
        self.callbacks.callback(iq)
        self.callbacks = None