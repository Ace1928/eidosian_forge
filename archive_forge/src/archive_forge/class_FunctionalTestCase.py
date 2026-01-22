import logging
import sys
from testtools import testcase
class FunctionalTestCase(TestCase):
    """Base for functional tests"""

    def setUp(self):
        super(FunctionalTestCase, self).setUp()
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)