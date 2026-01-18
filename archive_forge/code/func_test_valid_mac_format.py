from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_valid_mac_format(self):
    validate_format = ['01:23:45:67:89:ab', '01-23-45-67-89-ab', '0123.4567.89ab']
    for mac in validate_format:
        self.assertTrue(self.constraint.validate(mac, None))