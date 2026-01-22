from testtools import TestCase
from testtools.matchers import Contains
from fixtures import (
class TestByteStreams(TestCase):

    def test_empty_detail_stream(self):
        detail_name = 'test'
        fixture = ByteStream(detail_name)
        with fixture:
            content = fixture.getDetails()[detail_name]
            self.assertEqual('', content.as_text())

    def test_stream_content_in_details(self):
        detail_name = 'test'
        fixture = ByteStream(detail_name)
        with fixture:
            stream = fixture.stream
            content = fixture.getDetails()[detail_name]
            stream.write(b'testing 1 2 3')
            self.assertEqual('testing 1 2 3', content.as_text())

    def test_stream_content_reset(self):
        detail_name = 'test'
        fixture = ByteStream(detail_name)
        with fixture:
            stream = fixture.stream
            content = fixture.getDetails()[detail_name]
            stream.write(b'testing 1 2 3')
        with fixture:
            self.assertEqual('testing 1 2 3', content.as_text())
            content = fixture.getDetails()[detail_name]
            stream = fixture.stream
            stream.write(b'1 2 3 testing')
            self.assertEqual('1 2 3 testing', content.as_text())