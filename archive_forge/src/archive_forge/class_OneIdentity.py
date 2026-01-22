from datetime import datetime, timedelta
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
class OneIdentity(bakery.IdentityClient):
    """An IdentityClient implementation that always returns a single identity
    from declared_identity, allowing allow(LOGIN_OP) to work even when there
    are no declaration caveats (this is mostly to support the legacy tests
    which do their own checking of declaration caveats).
    """

    def identity_from_context(self, ctx):
        return (None, None)

    def declared_identity(self, ctx, declared):
        return _NoOne()