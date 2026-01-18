import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_repository_data_stream_error(self):
    """Test the formatting of RepositoryDataStreamError."""
    e = errors.RepositoryDataStreamError('my reason')
    self.assertEqual('Corrupt or incompatible data stream: my reason', str(e))