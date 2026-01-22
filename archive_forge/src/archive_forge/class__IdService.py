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
class _IdService(bakery.IdentityClient, bakery.ThirdPartyCaveatChecker):

    def __init__(self, location, locator, test_class):
        self._location = location
        self._test = test_class
        key = bakery.generate_key()
        self._discharger = _Discharger(key=key, checker=self, locator=locator)
        locator[location] = self._discharger

    def check_third_party_caveat(self, ctx, info):
        if info.condition != 'is-authenticated-user':
            raise bakery.CaveatNotRecognizedError('third party condition not recognized')
        username = ctx.get(_DISCHARGE_USER_KEY, '')
        if username == '':
            raise bakery.ThirdPartyCaveatCheckFailed('no current user')
        self._test._discharges.append(_DischargeRecord(location=self._location, user=username))
        return [checkers.declared_caveat('username', username)]

    def identity_from_context(self, ctx):
        return (None, [checkers.Caveat(location=self._location, condition='is-authenticated-user')])

    def declared_identity(self, ctx, declared):
        user = declared.get('username')
        if user is None:
            raise bakery.IdentityError('no username declared')
        return bakery.SimpleIdentity(user)