from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import pickle
import crcmod
import six
from six.moves import queue as Queue
from gslib.cs_api_map import ApiSelector
from gslib.parallel_tracker_file import ObjectFromTracker
from gslib.parallel_tracker_file import WriteParallelUploadTrackerFile
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import HaltingCopyCallbackHandler
from gslib.tests.util import HaltOneComponentCopyCallbackHandler
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import TailSet
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import TEST_ENCRYPTION_KEY2
from gslib.tests.util import unittest
from gslib.thread_message import FileMessage
from gslib.thread_message import FinalMessage
from gslib.thread_message import MetadataMessage
from gslib.thread_message import ProducerThreadMessage
from gslib.thread_message import ProgressMessage
from gslib.thread_message import SeekAheadMessage
from gslib.tracker_file import DeleteTrackerFile
from gslib.tracker_file import GetSlicedDownloadTrackerFilePaths
from gslib.tracker_file import GetTrackerFilePath
from gslib.tracker_file import TrackerFileType
from gslib.ui_controller import BytesToFixedWidthString
from gslib.ui_controller import DataManager
from gslib.ui_controller import MainThreadUIQueue
from gslib.ui_controller import MetadataManager
from gslib.ui_controller import UIController
from gslib.ui_controller import UIThread
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.constants import START_CALLBACK_PER_BYTES
from gslib.utils.constants import UTF8
from gslib.utils.copy_helper import PARALLEL_UPLOAD_STATIC_SALT
from gslib.utils.copy_helper import PARALLEL_UPLOAD_TEMP_NAMESPACE
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.parallelism_framework_util import PutToQueueWithTimeout
from gslib.utils.parallelism_framework_util import ZERO_TASKS_TO_DO_ARGUMENT
from gslib.utils.retry_util import Retry
from gslib.utils.unit_util import HumanReadableWithDecimalPlaces
from gslib.utils.unit_util import MakeHumanReadable
from gslib.utils.unit_util import ONE_KIB
def test_ui_throughput_calculation_with_no_components(self):
    """Tests throughput calculation in the UI.

    This test takes two different values, both with a different size and
    different number of components, and see if throughput behaves as expected.
    """
    status_queue = Queue.Queue()
    stream = six.StringIO()
    start_time = self.start_time
    ui_controller = UIController(sliding_throughput_period=2, update_message_period=1, first_throughput_latency=0, custom_time=start_time)
    ui_thread = UIThread(status_queue, stream, ui_controller)
    fpath1 = self.CreateTempFile(file_name='sample-file.txt', contents=b'foo')
    fpath2 = self.CreateTempFile(file_name='sample-file2.txt', contents=b'FOO')
    size1 = 1024 ** 2 * 60
    src_url1 = StorageUrlFromString(suri(fpath1))
    size2 = 1024 ** 2 * 600
    src_url2 = StorageUrlFromString(suri(fpath2))
    for file_message_type, operation_name in ((FileMessage.FILE_UPLOAD, 'Uploading'), (FileMessage.FILE_DOWNLOAD, 'Downloading')):
        PutToQueueWithTimeout(status_queue, FileMessage(src_url1, None, start_time + 200, size=size1, message_type=file_message_type))
        PutToQueueWithTimeout(status_queue, FileMessage(src_url2, None, start_time + 301, size=size2, message_type=file_message_type))
        progress_calls_number = 4
        for j in range(1, progress_calls_number + 1):
            PutToQueueWithTimeout(status_queue, ProgressMessage(size1, j * size1 / 4, src_url1, start_time + 300 + j * 2, operation_name=operation_name))
            PutToQueueWithTimeout(status_queue, ProgressMessage(size2, j * size2 / 4, src_url2, start_time + 300 + j * 2 + 1, operation_name=operation_name))
        PutToQueueWithTimeout(status_queue, FileMessage(src_url1, None, start_time + 700, size=size1, finished=True, message_type=file_message_type))
        PutToQueueWithTimeout(status_queue, FileMessage(src_url2, None, start_time + 800, size=size2, finished=True, message_type=file_message_type))
        PutToQueueWithTimeout(status_queue, ZERO_TASKS_TO_DO_ARGUMENT)
        JoinThreadAndRaiseOnTimeout(ui_thread)
        content = stream.getvalue()
        zero = BytesToFixedWidthString(0)
        self.assertIn(zero + '/s', content)
        file1_progress = size1 / progress_calls_number
        file2_progress = size2 / progress_calls_number
        average_progress = BytesToFixedWidthString((file1_progress + file2_progress) / 2)
        self.assertEqual(content.count(average_progress + '/s'), 2 * progress_calls_number - 1)