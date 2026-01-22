import unittest
from testtools import (
from testtools.compat import _b
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.testresult.doubles import (
class DetailsFixture(fixtures.Fixture):

    def setUp(self):
        fixtures.Fixture.setUp(self)
        self.addDetail('aaa', content.text_content('foo'))
        self.addDetail('bbb', content.text_content('bar'))