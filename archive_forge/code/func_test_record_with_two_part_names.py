from io import BytesIO
from ... import tests
from .. import pack
def test_record_with_two_part_names(self):
    """Reading a Bytes record with a two_part name reads both."""
    self.assertRecordParsing(([(b'name1', b'name2')], b'aaaaa'), b'5\nname1\x00name2\n\naaaaa')