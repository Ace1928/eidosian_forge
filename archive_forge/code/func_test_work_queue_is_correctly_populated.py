import tempfile
from boto.compat import Queue
from tests.compat import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.glacier.concurrent import ConcurrentUploader, ConcurrentDownloader
from boto.glacier.concurrent import UploadWorkerThread
from boto.glacier.concurrent import _END_SENTINEL
def test_work_queue_is_correctly_populated(self):
    uploader = FakeThreadedConcurrentUploader(mock.MagicMock(), 'vault_name')
    uploader.upload('foofile')
    q = uploader.worker_queue
    items = [q.get() for i in range(q.qsize())]
    self.assertEqual(items[0], (0, 4 * 1024 * 1024))
    self.assertEqual(items[1], (1, 4 * 1024 * 1024))
    self.assertEqual(len(items), 12)