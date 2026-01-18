from testtools import TestCase
from testtools.matchers import Contains
from fixtures import (
def test_stream_content_reset(self):
    detail_name = 'test'
    fixture = StringStream(detail_name)
    with fixture:
        stream = fixture.stream
        content = fixture.getDetails()[detail_name]
        stream.write('testing 1 2 3')
    with fixture:
        self.assertEqual('testing 1 2 3', content.as_text())
        content = fixture.getDetails()[detail_name]
        stream = fixture.stream
        stream.write('1 2 3 testing')
        self.assertEqual('1 2 3 testing', content.as_text())