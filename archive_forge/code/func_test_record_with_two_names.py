from io import BytesIO
from ... import tests
from .. import pack
def test_record_with_two_names(self):
    """Reading a Bytes record with two names returns a list of both names.
        """
    self.assertRecordParsing(([(b'name1',), (b'name2',)], b'aaaaa'), b'5\nname1\nname2\n\naaaaa')