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
def _discharge_all(self, ctx, m):

    def get_discharge(cav, payload):
        d = self._dischargers.get(cav.location)
        if d is None:
            raise ValueError('third party discharger {} not found'.format(cav.location))
        return d.discharge(ctx, cav, payload)
    return bakery.discharge_all(m, get_discharge)