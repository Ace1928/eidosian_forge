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
def test_ui_controller_shared_states(self):
    """Tests that UIController correctly integrates messages.

    This test ensures UIController correctly shares its state, which is used by
    both UIThread and MainThreadUIQueue. There are multiple ways of checking
    that. One such way is to create a ProducerThreadMessage on the
    MainThreadUIQueue, simulate a upload with messages coming from the UIThread,
    and check if the output has the percentage done and number of files
    (both happen only when a ProducerThreadMessage or SeekAheadMessage is
    called).
    """
    ui_thread_status_queue = Queue.Queue()
    stream = six.StringIO()
    start_time = self.start_time
    ui_controller = UIController(0, 0, 0, 0, custom_time=start_time)
    main_thread_ui_queue = MainThreadUIQueue(stream, ui_controller)
    ui_thread = UIThread(ui_thread_status_queue, stream, ui_controller)
    PutToQueueWithTimeout(main_thread_ui_queue, ProducerThreadMessage(1, UPLOAD_SIZE, start_time, finished=True))
    fpath = self.CreateTempFile(file_name='sample-file.txt', contents=b'foo')
    PutToQueueWithTimeout(ui_thread_status_queue, FileMessage(StorageUrlFromString(suri(fpath)), None, start_time + 10, size=UPLOAD_SIZE, message_type=FileMessage.FILE_UPLOAD, finished=False))
    PutToQueueWithTimeout(ui_thread_status_queue, FileMessage(StorageUrlFromString(suri(fpath)), None, start_time + 20, size=UPLOAD_SIZE, message_type=FileMessage.FILE_UPLOAD, finished=True))
    PutToQueueWithTimeout(ui_thread_status_queue, FinalMessage(start_time + 50))
    PutToQueueWithTimeout(ui_thread_status_queue, ZERO_TASKS_TO_DO_ARGUMENT)
    JoinThreadAndRaiseOnTimeout(ui_thread)
    content = stream.getvalue()
    CheckUiOutputWithMFlag(self, content, 1, UPLOAD_SIZE)