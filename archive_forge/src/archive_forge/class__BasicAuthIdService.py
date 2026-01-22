import base64
import json
from collections import namedtuple
from datetime import timedelta
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import pymacaroons
from macaroonbakery.tests.common import epoch, test_checker, test_context
from pymacaroons.verifier import FirstPartyCaveatVerifierDelegate, Verifier
class _BasicAuthIdService(bakery.IdentityClient):

    def identity_from_context(self, ctx):
        user, pwd = _basic_auth_from_context(ctx)
        if user != 'sherlock' or pwd != 'holmes':
            return (None, None)
        return (bakery.SimpleIdentity(user), None)

    def declared_identity(self, ctx, declared):
        raise bakery.IdentityError('no identity declarations in basic auth id service')