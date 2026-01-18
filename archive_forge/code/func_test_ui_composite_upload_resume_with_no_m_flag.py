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
@SkipForS3('No resumable upload support for S3.')
def test_ui_composite_upload_resume_with_no_m_flag(self):
    """Tests UI on a resumable upload with finished components and no -m flag.
    """
    self._test_ui_composite_upload_resume_helper()