import sys
import warnings
from oslo_log import log
from sqlalchemy import exc
from testtools import matchers
from keystone.tests import unit
class TestChild(TestParent):

    def test_in_parent(self):
        self.skip_test_overrides('some message')

    def test_not_in_parent(self):
        self.skip_test_overrides('some message')