from twisted.words.protocols.jabber import error, sasl, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish, utility, xpath
class BasicAuthenticator(xmlstream.ConnectAuthenticator):
    """
    Authenticates an XmlStream against a Jabber server as a Client.

    This only implements non-SASL authentication, per
    U{JEP-0078<http://www.jabber.org/jeps/jep-0078.html>}. Additionally, this
    authenticator provides the ability to perform inline registration, per
    U{JEP-0077<http://www.jabber.org/jeps/jep-0077.html>}.

    Under normal circumstances, the BasicAuthenticator generates the
    L{xmlstream.STREAM_AUTHD_EVENT} once the stream has authenticated. However,
    it can also generate other events, such as:
      - L{INVALID_USER_EVENT} : Authentication failed, due to invalid username
      - L{AUTH_FAILED_EVENT} : Authentication failed, due to invalid password
      - L{REGISTER_FAILED_EVENT} : Registration failed

    If authentication fails for any reason, you can attempt to register by
    calling the L{registerAccount} method. If the registration succeeds, a
    L{xmlstream.STREAM_AUTHD_EVENT} will be fired. Otherwise, one of the above
    errors will be generated (again).


    @cvar INVALID_USER_EVENT: See L{IQAuthInitializer.INVALID_USER_EVENT}.
    @type INVALID_USER_EVENT: L{str}

    @cvar AUTH_FAILED_EVENT: See L{IQAuthInitializer.AUTH_FAILED_EVENT}.
    @type AUTH_FAILED_EVENT: L{str}

    @cvar REGISTER_FAILED_EVENT: Token to signal that registration failed.
    @type REGISTER_FAILED_EVENT: L{str}

    """
    namespace = 'jabber:client'
    INVALID_USER_EVENT = IQAuthInitializer.INVALID_USER_EVENT
    AUTH_FAILED_EVENT = IQAuthInitializer.AUTH_FAILED_EVENT
    REGISTER_FAILED_EVENT = '//event/client/basicauth/registerfailed'

    def __init__(self, jid, password):
        xmlstream.ConnectAuthenticator.__init__(self, jid.host)
        self.jid = jid
        self.password = password

    def associateWithStream(self, xs):
        xs.version = (0, 0)
        xmlstream.ConnectAuthenticator.associateWithStream(self, xs)
        xs.initializers = [xmlstream.TLSInitiatingInitializer(xs, required=False), IQAuthInitializer(xs)]

    def registerAccount(self, username=None, password=None):
        if username:
            self.jid.user = username
        if password:
            self.password = password
        iq = IQ(self.xmlstream, 'set')
        iq.addElement(('jabber:iq:register', 'query'))
        iq.query.addElement('username', content=self.jid.user)
        iq.query.addElement('password', content=self.password)
        iq.addCallback(self._registerResultEvent)
        iq.send()

    def _registerResultEvent(self, iq):
        if iq['type'] == 'result':
            self.streamStarted()
        else:
            self.xmlstream.dispatch(iq, self.REGISTER_FAILED_EVENT)