import abc
from ._error import IdentityError
class IdentityClient(object):
    """ Represents an abstract identity manager. User identities can be based
    on local informaton (for example HTTP basic auth) or by reference to an
    external trusted third party (an identity manager).
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def identity_from_context(self, ctx):
        """ Returns the identity based on information in the context.

        If it cannot determine the identity based on the context, then it
        should return a set of caveats containing a third party caveat that,
        when discharged, can be used to obtain the identity with
        declared_identity.

        It should only raise an error if it cannot check the identity
        (for example because of a database access error) - it's
        OK to return all zero values when there's
        no identity found and no third party to address caveats to.
        @param ctx an AuthContext
        :return: an Identity and array of caveats
        """
        raise NotImplementedError('identity_from_context method must be defined in subclass')

    @abc.abstractmethod
    def declared_identity(self, ctx, declared):
        """Parses the identity declaration from the given declared attributes.

        TODO take the set of first party caveat conditions instead?
        @param ctx (AuthContext)
        @param declared (dict of string/string)
        :return: an Identity
        """
        raise NotImplementedError('declared_identity method must be defined in subclass')