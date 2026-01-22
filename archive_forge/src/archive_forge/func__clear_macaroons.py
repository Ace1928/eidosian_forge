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
def _clear_macaroons(self, svc):
    if svc is None:
        self._macaroons = {}
        return
    if svc.name() in self._macaroons:
        del self._macaroons[svc.name()]