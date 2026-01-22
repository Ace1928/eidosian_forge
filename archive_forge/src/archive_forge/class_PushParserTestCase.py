from io import BytesIO
from ... import tests
from .. import pack
class PushParserTestCase(tests.TestCase):
    """Base class for TestCases involving ContainerPushParser."""

    def make_parser_expecting_record_type(self):
        parser = pack.ContainerPushParser()
        parser.accept_bytes(b'Bazaar pack format 1 (introduced in 0.18)\n')
        return parser

    def make_parser_expecting_bytes_record(self):
        parser = pack.ContainerPushParser()
        parser.accept_bytes(b'Bazaar pack format 1 (introduced in 0.18)\nB')
        return parser

    def assertRecordParsing(self, expected_record, data):
        """Assert that 'bytes' is parsed as a given bytes record.

        :param expected_record: A tuple of (names, bytes).
        """
        parser = self.make_parser_expecting_bytes_record()
        parser.accept_bytes(data)
        parsed_records = parser.read_pending_records()
        self.assertEqual([expected_record], parsed_records)