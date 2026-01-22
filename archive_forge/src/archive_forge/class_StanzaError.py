import copy
from typing import Optional
from twisted.words.xish import domish
class StanzaError(BaseError):
    """
    Stanza Error exception.

    Refer to RFC 3920, section 9.3, for the allowed values for C{condition} and
    C{type}.

    @ivar type: The stanza error type. Gives a suggestion to the recipient
                of the error on how to proceed.
    @type type: C{str}
    @ivar code: A numeric identifier for the error condition for backwards
                compatibility with pre-XMPP Jabber implementations.
    """
    namespace = NS_XMPP_STANZAS

    def __init__(self, condition, type=None, text=None, textLang=None, appCondition=None):
        BaseError.__init__(self, condition, text, textLang, appCondition)
        if type is None:
            try:
                type = STANZA_CONDITIONS[condition]['type']
            except KeyError:
                pass
        self.type = type
        try:
            self.code = STANZA_CONDITIONS[condition]['code']
        except KeyError:
            self.code = None
        self.children = []
        self.iq = None

    def getElement(self):
        """
        Get XML representation from self.

        Overrides the base L{BaseError.getElement} to make sure the returned
        element has a C{type} attribute and optionally a legacy C{code}
        attribute.

        @rtype: L{domish.Element}
        """
        error = BaseError.getElement(self)
        error['type'] = self.type
        if self.code:
            error['code'] = self.code
        return error

    def toResponse(self, stanza):
        """
        Construct error response stanza.

        The C{stanza} is transformed into an error response stanza by
        swapping the C{to} and C{from} addresses and inserting an error
        element.

        @note: This creates a shallow copy of the list of child elements of the
               stanza. The child elements themselves are not copied themselves,
               and references to their parent element will still point to the
               original stanza element.

               The serialization of an element does not use the reference to
               its parent, so the typical use case of immediately sending out
               the constructed error response is not affected.

        @param stanza: the stanza to respond to
        @type stanza: L{domish.Element}
        """
        from twisted.words.protocols.jabber.xmlstream import toResponse
        response = toResponse(stanza, stanzaType='error')
        response.children = copy.copy(stanza.children)
        response.addChild(self.getElement())
        return response