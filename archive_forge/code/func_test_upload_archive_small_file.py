from boto.compat import StringIO
from tests.compat import mock, unittest
from boto.glacier import vault
from boto.glacier.job import Job
from boto.glacier.response import GlacierResponse
@mock.patch('boto.glacier.vault.compute_hashes_from_fileobj', return_value=[b'abc', b'123'])
def test_upload_archive_small_file(self, compute_hashes):
    self.getsize.return_value = 1
    self.api.upload_archive.return_value = {'ArchiveId': 'archive_id'}
    with mock.patch('boto.glacier.vault.open', self.mock_open, create=True):
        archive_id = self.vault.upload_archive('filename', 'my description')
    self.assertEqual(archive_id, 'archive_id')
    self.api.upload_archive.assert_called_with('myvault', self.mock_open.return_value, mock.ANY, mock.ANY, 'my description')