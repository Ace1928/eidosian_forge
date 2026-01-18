from ... import tests
from .. import generate_ids
def test_revision_ids_are_ascii(self):
    """gen_revision_id should always return an ascii revision id."""
    tail = b'-\\d{14}-[a-z0-9]{16}'
    self.assertGenRevisionId(b'joe_bar' + tail, 'Joe Bar')
    self.assertGenRevisionId(b'joe_bar' + tail, 'Joe Bar')
    self.assertGenRevisionId(b'joe@foo' + tail, 'Joe Bar <joe@foo>')
    self.assertGenRevisionId(b'joe@f' + tail, 'Joe Bar <joe@f¶>')