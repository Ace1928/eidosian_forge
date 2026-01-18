import os
import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons import MACAROON_V1, Macaroon
 Runs a similar test as test_macaroon_paper_fig6 with the discharge
        macaroon binding being done on a tampered signature.
        