import io
import logging
import time
import testtools
from testtools import TestCase
from fixtures import (
def test_output_property_has_output(self):
    fixture = self.useFixture(FakeLogger())
    logging.info('some message')
    self.assertEqual('some message\n', fixture.output)