import copy
import math
from s3transfer.tasks import Task
from s3transfer.tasks import SubmissionTask
from s3transfer.tasks import CreateMultipartUploadTask
from s3transfer.tasks import CompleteMultipartUploadTask
from s3transfer.utils import get_callbacks
from s3transfer.utils import calculate_range_parameter
from s3transfer.utils import get_filtered_dict
from s3transfer.utils import ChunksizeAdjuster
def _get_head_object_request_from_copy_source(self, copy_source):
    if isinstance(copy_source, dict):
        return copy.copy(copy_source)
    else:
        raise TypeError('Expecting dictionary formatted: {"Bucket": bucket_name, "Key": key} but got %s or type %s.' % (copy_source, type(copy_source)))