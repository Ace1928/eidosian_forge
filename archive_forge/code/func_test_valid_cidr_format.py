from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_valid_cidr_format(self):
    validate_format = ['10.0.0.0/24', '6000::/64']
    for cidr in validate_format:
        self.assertTrue(self.constraint.validate(cidr, None))