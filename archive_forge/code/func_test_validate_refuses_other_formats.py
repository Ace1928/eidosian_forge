from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_validate_refuses_other_formats(self):
    self.assertFalse(self.constraint.validate('Fri 13th, 2050', None))