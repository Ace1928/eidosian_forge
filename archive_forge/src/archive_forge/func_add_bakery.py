import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons.verifier import Verifier
def add_bakery():
    M.bakery_id += 1
    loc = 'loc{}'.format(M.bakery_id)
    bakeries[loc] = common.new_bakery(loc, locator)
    return loc