import errno
from .. import osutils, tests
from . import features
def test_top_prefix_to_starting_dir(self):
    self.assertEqual(('prefix', None, None, None, '\x12'), self.reader.top_prefix_to_starting_dir('\x12'.encode(), 'prefix'))