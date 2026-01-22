import abc
from ._error import IdentityError
class NoIdentities(IdentityClient):
    """ Defines the null identity provider - it never returns any identities.
    """

    def identity_from_context(self, ctx):
        return (None, None)

    def declared_identity(self, ctx, declared):
        raise IdentityError('no identity declared or possible')