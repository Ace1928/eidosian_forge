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
def _resolve_caveats(ns, caveats):
    conds = [''] * len(caveats)
    for i, cav in enumerate(caveats):
        if cav.location is not None and cav.location != '':
            raise ValueError('found unexpected third party caveat')
        conds[i] = ns.resolve_caveat(cav).condition
    return conds