from testtools import TestCase
from testtools.matchers import Equals, MatchesException, Raises
from testtools.content_type import (
class TestBuiltinContentTypes(TestCase):

    def test_plain_text(self):
        self.assertThat(UTF8_TEXT.type, Equals('text'))
        self.assertThat(UTF8_TEXT.subtype, Equals('plain'))
        self.assertThat(UTF8_TEXT.parameters, Equals({'charset': 'utf8'}))

    def test_json_content(self):
        self.assertThat(JSON.type, Equals('application'))
        self.assertThat(JSON.subtype, Equals('json'))
        self.assertThat(JSON.parameters, Equals({}))