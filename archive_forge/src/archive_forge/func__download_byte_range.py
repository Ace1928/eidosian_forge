import math
import socket
from boto.glacier.exceptions import TreeHashDoesNotMatchError, \
from boto.glacier.utils import tree_hash_from_str
def _download_byte_range(self, byte_range, retry_exceptions):
    for _ in range(5):
        try:
            response = self.get_output(byte_range)
            data = response.read()
            expected_tree_hash = response['TreeHash']
            return (data, expected_tree_hash)
        except retry_exceptions as e:
            continue
    else:
        raise DownloadArchiveError('There was an error downloadingbyte range %s: %s' % (byte_range, e))