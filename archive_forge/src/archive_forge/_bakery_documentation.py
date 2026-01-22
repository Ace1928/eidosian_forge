from ._authorizer import ClosedAuthorizer
from ._checker import Checker
import macaroonbakery.checkers as checkers
from ._oven import Oven
Returns a new Bakery instance which combines an Oven with a
        Checker for the convenience of callers that wish to use both
        together.
        @param checker holds the checker used to check first party caveats.
        If this is None, it will use checkers.Checker(None).
        @param root_key_store holds the root key store to use.
        If you need to use a different root key store for different operations,
        you'll need to pass a root_key_store_for_ops value to Oven directly.
        @param root_key_store If this is None, it will use MemoryKeyStore().
        Note that that is almost certain insufficient for production services
        that are spread across multiple instances or that need
        to persist keys across restarts.
        @param locator is used to find out information on third parties when
        adding third party caveats. If this is None, no non-local third
        party caveats can be added.
        @param key holds the private key of the oven. If this is None,
        no third party caveats may be added.
        @param identity_client holds the identity implementation to use for
        authentication. If this is None, no authentication will be possible.
        @param authorizer is used to check whether an authenticated user is
        allowed to perform operations. If it is None, it will use
        a ClosedAuthorizer.
        The identity parameter passed to authorizer.allow will
        always have been obtained from a call to
        IdentityClient.declared_identity.
        @param ops_store used to persistently store the association of
        multi-op entities with their associated operations
        when oven.macaroon is called with multiple operations.
        @param location holds the location to use when creating new macaroons.
        