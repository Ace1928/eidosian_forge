from testtools import TestCase
from testtools.matchers import Equals, MatchesException, Raises
from testtools.content_type import (
def test_plain_text(self):
    self.assertThat(UTF8_TEXT.type, Equals('text'))
    self.assertThat(UTF8_TEXT.subtype, Equals('plain'))
    self.assertThat(UTF8_TEXT.parameters, Equals({'charset': 'utf8'}))