from twisted.words.protocols.jabber import error, sasl, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish, utility, xpath
class XMPPAuthenticator(xmlstream.ConnectAuthenticator):
    """
    Initializes an XmlStream connecting to an XMPP server as a Client.

    This authenticator performs the initialization steps needed to start
    exchanging XML stanzas with an XMPP server as an XMPP client. It checks if
    the server advertises XML stream version 1.0, negotiates TLS (when
    available), performs SASL authentication, binds a resource and establishes
    a session.

    Upon successful stream initialization, the L{xmlstream.STREAM_AUTHD_EVENT}
    event will be dispatched through the XML stream object. Otherwise, the
    L{xmlstream.INIT_FAILED_EVENT} event will be dispatched with a failure
    object.

    After inspection of the failure, initialization can then be restarted by
    calling L{ConnectAuthenticator.initializeStream}. For example, in case of
    authentication failure, a user may be given the opportunity to input the
    correct password.  By setting the L{password} instance variable and restarting
    initialization, the stream authentication step is then retried, and subsequent
    steps are performed if successful.

    @ivar jid: Jabber ID to authenticate with. This may contain a resource
               part, as a suggestion to the server for resource binding. A
               server may override this, though. If the resource part is left
               off, the server will generate a unique resource identifier.
               The server will always return the full Jabber ID in the
               resource binding step, and this is stored in this instance
               variable.
    @type jid: L{jid.JID}

    @ivar password: password to be used during SASL authentication.
    @type password: L{unicode}
    """
    namespace = 'jabber:client'

    def __init__(self, jid, password, configurationForTLS=None):
        """
        @param configurationForTLS: An object which creates appropriately
            configured TLS connections. This is passed to C{startTLS} on the
            transport and is preferably created using
            L{twisted.internet.ssl.optionsForClientTLS}. If C{None}, the
            default is to verify the server certificate against the trust roots
            as provided by the platform. See
            L{twisted.internet._sslverify.platformTrust}.
        @type configurationForTLS: L{IOpenSSLClientConnectionCreator} or
            C{None}
        """
        xmlstream.ConnectAuthenticator.__init__(self, jid.host)
        self.jid = jid
        self.password = password
        self._configurationForTLS = configurationForTLS

    def associateWithStream(self, xs):
        """
        Register with the XML stream.

        Populates stream's list of initializers, along with their
        requiredness. This list is used by
        L{ConnectAuthenticator.initializeStream} to perform the initialization
        steps.
        """
        xmlstream.ConnectAuthenticator.associateWithStream(self, xs)
        xs.initializers = [CheckVersionInitializer(xs), xmlstream.TLSInitiatingInitializer(xs, required=True, configurationForTLS=self._configurationForTLS), sasl.SASLInitiatingInitializer(xs, required=True), BindInitializer(xs, required=True), SessionInitializer(xs, required=False)]