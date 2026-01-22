from zope.interface import Interface
class IMessageDelivery(Interface):

    def receivedHeader(helo, origin, recipients):
        """
        Generate the Received header for a message.

        @type helo: 2-L{tuple} of L{bytes} and L{bytes}.
        @param helo: The argument to the HELO command and the client's IP
        address.

        @type origin: L{Address}
        @param origin: The address the message is from

        @type recipients: L{list} of L{User}
        @param recipients: A list of the addresses for which this message
        is bound.

        @rtype: L{bytes}
        @return: The full C{"Received"} header string.
        """

    def validateTo(user):
        """
        Validate the address for which the message is destined.

        @type user: L{User}
        @param user: The address to validate.

        @rtype: no-argument callable
        @return: A L{Deferred} which becomes, or a callable which takes no
            arguments and returns an object implementing L{IMessageSMTP}. This
            will be called and the returned object used to deliver the message
            when it arrives.

        @raise SMTPBadRcpt: Raised if messages to the address are not to be
            accepted.
        """

    def validateFrom(helo, origin):
        """
        Validate the address from which the message originates.

        @type helo: 2-L{tuple} of L{bytes} and L{bytes}.
        @param helo: The argument to the HELO command and the client's IP
        address.

        @type origin: L{Address}
        @param origin: The address the message is from

        @rtype: L{Deferred} or L{Address}
        @return: C{origin} or a L{Deferred} whose callback will be
        passed C{origin}.

        @raise SMTPBadSender: Raised of messages from this address are
        not to be accepted.
        """