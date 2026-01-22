from zope.interface import Interface
class IDomain(Interface):
    """
    An interface for email domains.
    """

    def exists(user):
        """
        Check whether a user exists in this domain.

        @type user: L{User}
        @param user: A user.

        @rtype: no-argument callable which returns L{IMessageSMTP} provider
        @return: A function which takes no arguments and returns a message
            receiver for the user.

        @raise SMTPBadRcpt: When the given user does not exist in this domain.
        """

    def addUser(user, password):
        """
        Add a user to this domain.

        @type user: L{bytes}
        @param user: A username.

        @type password: L{bytes}
        @param password: A password.
        """

    def getCredentialsCheckers():
        """
        Return credentials checkers for this domain.

        @rtype: L{list} of L{ICredentialsChecker
            <twisted.cred.checkers.ICredentialsChecker>} provider
        @return: Credentials checkers for this domain.
        """