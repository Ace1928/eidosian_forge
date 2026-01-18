from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import pkgutil
from six.moves import range
from gslib.exception import CommandException
from gslib.resumable_streaming_upload import ResumableStreamingJsonUploadWrapper
import gslib.tests.testcase as testcase
from gslib.utils.boto_util import GetJsonResumableChunkSize
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
from gslib.utils.hashing_helper import CalculateHashesFromContents
from gslib.utils.hashing_helper import CalculateMd5FromContents
from gslib.utils.hashing_helper import GetMd5
Tests seeking back partially within the buffer.